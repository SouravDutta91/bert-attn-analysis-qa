import collections
import torch
import torch.nn.functional as F


class BertExplainer:
    def __init__(self, model):
        self.model = model
        self.layer_values_global = collections.OrderedDict()

    @staticmethod
    def save_inputs(self, layer_values, module_name):
        def _save_inputs(module, input, output):
            layer_values[module_name] = {"input": input, "output": output}

        return _save_inputs

    def register_hooks(self, model, layer_values, parent_name=""):
        modules = list(model.named_modules())
        for i, (name, module) in enumerate(modules):
            module.register_forward_hook(self.save_inputs(layer_values, name))
            # module.register_backward_hook(compute_relevance(layer_values, name))

        return layer_values

    @staticmethod
    def compute_matmul_relevance(self, input1, input2, relevance):
        input1_relevance = torch.zeros(input1.shape, dtype=torch.float, device="cuda")
        input2_relevance = torch.zeros(input2.shape, dtype=torch.float, device="cuda")
        for i in range(input1.shape[0]):
            for j in range(input1.shape[1]):
                out = torch.matmul(input1[i][j], input2[i][j])
                presum = input1[i][j].unsqueeze(-1) * input2[i][j]
                contributions = relevance[i][j].unsqueeze(-2) * presum / out.unsqueeze(-2)
                input1_relevance[i][j] = contributions.sum(-1)
                input2_relevance[i][j] = contributions.sum(0)
        # print("input1_relevance "+str(input1_relevance.sum()))
        # print("input2_relevance "+str(input2_relevance.sum()))

        return input1_relevance, input2_relevance

    @staticmethod
    def compute_input_relevance(self, weight, input, relevance):
        # print("compute_input_relevance begin")
        input_relevance = torch.zeros(input.shape, dtype=torch.float, device="cuda")
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                input_contributions = weight * input[i][j].unsqueeze(1)
                input_relevance[i][j] = (input_contributions * relevance[i][j].unsqueeze(0)).sum(-1)
                # print(input_relevance.sum())
        # print("input_relevance "+str(input_relevance.sum()))
        input_relevance_normalized = input_relevance / input_relevance.sum((-2, -1)).unsqueeze(-1).unsqueeze(-1)
        # print("input_relevance_normalized "+str(input_relevance_normalized.sum()))
        # print("compute_input_relevance end")

        return input_relevance_normalized

    def compute_linear_3(self, module, module_name, relevance):
        # print("compute_linear_3 begin")
        # print("relevance "+str(relevance.sum()))
        input = self.layer_values_global[module_name]["input"][0]
        weight = module._parameters["weight"].transpose(-2, -1)
        output = torch.matmul(input, weight) + module._parameters["bias"]
        relevance *= (output - module._parameters["bias"]) / output
        # print("((output-module._parameters[\"bias\"])/output).sum() "+str(((output-module._parameters["bias"])/output).sum()))
        # print("relevance "+str(relevance.sum()))
        input_relevance = self.compute_input_relevance(weight, input, relevance)
        # print("input_relevance "+str(input_relevance.sum()))
        # print("compute_linear_3 end")

        return input_relevance

    def compute_bert_output(self, module, module_name, relevance):
        # print("compute_bert_output relevance "+str(relevance.sum()))
        _, input_tensor = self.layer_values_global[module_name]["input"]
        hidden_states = self.layer_values_global[module_name + ".dropout"]["input"][0]
        layer_norm_relevance = relevance
        hidden_states_relevance = layer_norm_relevance.clone().detach()
        input_tensor_relevance = layer_norm_relevance.clone().detach()
        # print("input_tensor_relevance "+str(input_tensor_relevance.sum()))
        hidden_states_relevance = self.compute_linear_3(module.dense, module_name + ".dense", hidden_states_relevance)
        # print("input_tensor_relevance "+str(input_tensor_relevance.sum()))

        return hidden_states_relevance, input_tensor_relevance

    def compute_self_attention(self, module, module_name, relevance):
        # print("relevance "+str(relevance.sum()))
        relevance = module.transpose_for_scores(relevance)
        hidden_states, attention_mask = self.layer_values_global[module_name]["input"]
        attention_probs = self.layer_values_global[module_name + ".dropout"]["input"][0]
        value_layer = module.transpose_for_scores(self.layer_values_global[module_name + ".value"]["output"])
        attention_probs_relevance, value_layer_relevance = self.compute_matmul_relevance(attention_probs, value_layer,
                                                                                         relevance)
        # print("attention_probs_relevance "+str(attention_probs_relevance.sum()))
        # print("value_layer_relevance "+str(value_layer_relevance.sum()))

        query_layer = module.transpose_for_scores(self.layer_values_global[module_name + ".query"]["output"])
        key_layer = module.transpose_for_scores(self.layer_values_global[module_name + ".key"]["output"])
        query_layer_relevance, key_layer_relevance = self.compute_matmul_relevance(query_layer,
                                                                                   key_layer.transpose(-1, -2),
                                                                                   attention_probs_relevance)
        key_layer_relevance = key_layer_relevance.transpose(-1, -2)
        # print("query_layer_relevance "+str(query_layer_relevance.sum()))
        # print("key_layer_relevance "+str(key_layer_relevance.sum()))

        new_shape = (query_layer_relevance.size()[0], query_layer_relevance.size()[2]) + (module.all_head_size,)
        query_layer_relevance = query_layer_relevance.permute(0, 2, 1, 3).contiguous().view(new_shape)
        key_layer_relevance = key_layer_relevance.permute(0, 2, 1, 3).contiguous().view(new_shape)
        value_layer_relevance = value_layer_relevance.permute(0, 2, 1, 3).contiguous().view(new_shape)

        hidden_state_relevance1 = self.compute_linear_3(module.key, module_name + ".key", key_layer_relevance)
        hidden_state_relevance2 = self.compute_linear_3(module.value, module_name + ".value", value_layer_relevance)
        hidden_state_relevance3 = self.compute_linear_3(module.query, module_name + ".query", query_layer_relevance)
        # print("hidden_state_relevance1 "+str(hidden_state_relevance1.sum()))
        # print("hidden_state_relevance2 "+str(hidden_state_relevance2.sum()))
        # print("hidden_state_relevance3 "+str(hidden_state_relevance3.sum()))

        return (hidden_state_relevance1 + hidden_state_relevance2 + hidden_state_relevance3) / 3

    def compute_attention(self, module, module_name, relevance):
        # print("compute_attention relevance "+str(relevance.sum()))
        bert_output_hidden_relevance, bert_output_input_relevance = self.compute_bert_output(module.output,
                                                                                             module_name + ".output",
                                                                                             relevance)
        # print("bert_output_input_relevance "+str(bert_output_input_relevance.sum()))
        attention_relevance = self.compute_self_attention(module.self, module_name + ".self",
                                                          bert_output_hidden_relevance)
        # print("attention_relevance "+str(attention_relevance.sum()))

        return (attention_relevance + bert_output_input_relevance) / 2  # FIX IT

    def compute_bert_layer(self, module, module_name, relevance):
        bert_output_hidden_relevance, bert_output_input_relevance = self.compute_bert_output(module.output,
                                                                                             module_name + ".output",
                                                                                             relevance)
        intermediate_relevance = self.compute_linear_3(module.intermediate.dense, module_name + ".intermediate",
                                                       bert_output_hidden_relevance)
        # print("intermediate_relevance "+str(intermediate_relevance.sum()))
        attention_relevance = self.compute_attention(module.attention, module_name + ".attention", (
                    intermediate_relevance + bert_output_input_relevance) / 2)  # FIX IT
        # print("attention_relevance "+str(attention_relevance.sum()))

        return attention_relevance

    def explain(self, input_ids, segment_ids, input_mask, target_outputs):
        self.register_hooks(self.model, self.layer_values_global)
        self.model = self.model.eval()
        with torch.no_grad():
            batch_start_logits, batch_end_logits = self.model(input_ids, segment_ids, input_mask)
            batch_start_logits_softmax = F.softmax(batch_start_logits, dim=1)
            batch_end_logits_softmax = F.softmax(batch_end_logits, dim=1)
            self.relevance_start_logits = torch.zeros(batch_start_logits.shape, dtype=torch.float, device="cuda")
            self.relevance_end_logits = torch.zeros(batch_end_logits.shape, dtype=torch.float, device="cuda")
            for i, output_neuron in enumerate(target_outputs):
                self.relevance_start_logits[i][output_neuron[0]] = 0.5
                self.relevance_end_logits[i][output_neuron[1]] = 0.5
            # loss = torch.sum(self.relevance_start_logits*batch_start_logits_softmax+self.relevance_end_logits*batch_end_logits_softmax)
            loss = torch.sum(batch_start_logits + batch_end_logits)
            # loss.register_backward_hook(compute_intermediate(layer_values, name))
            # loss.backward()

            qa_outputs_relevance = self.compute_linear_3(self.model.qa_outputs, "qa_outputs", torch.stack(
                (self.relevance_start_logits, self.relevance_end_logits), -1))
            # print("qa_outputs_relevance "+str(qa_outputs_relevance.sum()))
            encoder_layers_next_relevance = [qa_outputs_relevance]
            attentions = []
            self_attentions = []
            for i, layer_module in reversed(list(enumerate(self.model.bert.encoder.layer))):
                encoder_layers_next_relevance.append(
                    self.compute_bert_layer(layer_module, "bert.encoder.layer." + str(i),
                                            encoder_layers_next_relevance[-1]))
                attentions.append(self.layer_values_global["bert.encoder.layer." + str(i) + ".attention"]["output"])
                self_attentions.append(
                    self.layer_values_global["bert.encoder.layer." + str(i) + ".attention.self.dropout"]["input"])
                # print("encoder_layers_next_relevance "+str(encoder_layers_next_relevance[-1].sum((-1, -2))))

        return encoder_layers_next_relevance, attentions, self_attentions
