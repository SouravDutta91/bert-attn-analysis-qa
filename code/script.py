import math
import collections
import json
import pprint
# import simplejson                                         # TODO: check if needed

import torch
from torch import nn, optim
from torch.utils.data import (DataLoader,
                              SequentialSampler,
                              TensorDataset)

from tqdm import tqdm
from termcolor import colored

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering, BertConfig

from autoencoder import EncoderRNN, DecoderRNN, train_autoencoder
from squad import read_squad_examples, convert_examples_to_features, predict
from bert_explainer import BertExplainer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

# para_file = "../Input_file.txt"
para_file = "/content/drive/My Drive/train-v2.0.json"       # TODO: use proper file path
model_path = "/content/drive/My Drive/pytorch_model.bin"    # TODO: use proper file path

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

### Loading Pretrained model for QnA

config = BertConfig("../Results/bert_config.json")
model = BertForQuestionAnswering(config)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
# model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
model.to(device)
print()


### initializing the autoencoder

hidden_size = 384
encoder1 = EncoderRNN(384, config.hidden_size, hidden_size).to(device)
decoder1 = DecoderRNN(384, config.hidden_size, hidden_size).to(device)
encoder_optimizer = optim.Adam(encoder1.parameters())
decoder_optimizer = optim.Adam(decoder1.parameters())
criterion = nn.MSELoss()

pp = pprint.PrettyPrinter(indent=4)

# input_data is a list of dictionary which has a paragraph and questions
with open("/content/drive/My Drive/train-v2.0.json") as f:
    squad = json.load(f)
    for article in squad["data"]:
        # input_data = []
        # i = 1
        for context_questions in article["paragraphs"]:

            input_data = []
            i = 1

            paragraphs = {"id": i, "text": context_questions["context"]}
            paragraphs["ques"] = [(x["question"], x["is_impossible"]) for x in context_questions["qas"]]
            input_data.append(paragraphs)
            i += 1

            examples = read_squad_examples(input_data)

            eval_features = convert_examples_to_features(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=384,
                doc_stride=128,
                max_query_length=64)

            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

            pred_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
            # Run prediction for full data
            pred_sampler = SequentialSampler(pred_data)
            pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=9)

            predictions = []
            for input_ids, input_mask, segment_ids, example_indices in tqdm(pred_dataloader):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)

                # explainer = shap.DeepExplainer(model, [input_ids, segment_ids, input_mask])
                with torch.no_grad():
                    # tensor_output = model(input_ids, segment_ids, input_mask)
                    batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
                    # batch_start_logits, batch_end_logits = torch.split(tensor_output, int(tensor_output.shape[1]/2), dim=1)
                    # shap_values = explainer.shap_values([input_ids, segment_ids, input_mask])

                features = []
                examples_batch = []
                all_results = []

                print(len(examples), example_indices.max())
                for i, example_index in enumerate(example_indices):
                    start_logits = batch_start_logits[i].detach().cpu().tolist()
                    end_logits = batch_end_logits[i].detach().cpu().tolist()
                    feature = eval_features[example_index.item()]
                    unique_id = int(feature.unique_id)
                    features.append(feature)
                    examples_batch.append(examples[example_index.item()])
                    all_results.append(RawResult(unique_id=unique_id,
                                                 start_logits=start_logits,
                                                 end_logits=end_logits))

                output_indices = predict(examples_batch, features, all_results, 30)
                predictions.append(output_indices)

                explainer = BertExplainer(model)
                relevance, attentions, self_attentions = explainer.explain(input_ids, segment_ids, input_mask,
                                                                           [o["span"] for o in output_indices.values()])
                input_tensor = torch.stack(
                    [r.sum(-1).unsqueeze(-1) * explainer.layer_values_global["bert.encoder"]["input"][0] for r in
                     relevance], 0)
                target_tensor = torch.stack(relevance, 0).sum(-1)
                loss = train_autoencoder(input_tensor, target_tensor, encoder1,
                                         decoder1, encoder_optimizer, decoder_optimizer, criterion, max_length=13)

                print('Encoder loss: %.4f' % loss)

            # For printing the results ####
            index = None
            for example in examples:
                if index != example.example_id:
                    pp.pprint(example.para_text)
                    index = example.example_id
                    print('\n')
                    print(colored('***********Question and Answers *************', 'red'))

                ques_text = colored(example.question_text + " Unanswerable: " + str(example.unanswerable), 'blue')
                print(ques_text)
                prediction = colored(predictions[math.floor(example.unique_id / 9)][example]['text'], 'green',
                                     attrs=['reverse', 'blink'])
                print(prediction)
                print('\n')
