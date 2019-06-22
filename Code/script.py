from bertviz.attention_details import AttentionDetailsData, show
from bertviz.pytorch_pretrained_bert import BertModel, BertTokenizer

bert_version = 'bert-base-uncased'
model = BertModel.from_pretrained(bert_version)
tokenizer = BertTokenizer.from_pretrained(bert_version)

sentence_a = 'The cat sat on the mat'
sentence_b = 'The cat lay on the rug'

details_data = AttentionDetailsData(model, tokenizer)
tokens_a, tokens_b, queries, keys, atts = details_data.get_data(sentence_a, sentence_b)

show(tokens_a, tokens_b, queries, keys, atts)
