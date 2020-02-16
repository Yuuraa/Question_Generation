from numpy import argmax
from torch.nn.functional import softmax
from keras.preprocessing.sequence import pad_sequences
import torch

class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        # output 나온 것에 Wx+b 수행한 것
        logits = self.qa_outputs(sequence_output)
        # label의 갯수 만큼 logits를 쪼갤 것이다
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

class BertQuestionGenerator(BertModel):
    def __init__(self, dropout=0.9):
        super(BertQuestionGenerator,self).__init__()
        
        self.bert = get_kobert_model()
        self.qg_outputs = torch.nn.Linear(config.hiden_size, config.num_labels)
        # num_labels = 2, start 또는 end 가 될 수 있는 토큰 두 개
        self.linear = torch.nn.Linear(768,1)
        self.sigmoid = torch.nn.Sigmoid()
        self.apply(self.init_weights)
            
        def forward(self,input_ids, token_type_ids=None, attention_mask=None, start_positions=None,
            end_posit12ions=None, position_ids=None, head_mask=None):
                sequence_output,_ = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                        attention_mask=attention_mask, head_mask=head_mask)
                # output is a tuple whose contents depend on the task
                # sequence_output is encodings for all the elements in the input sequence
                    
                # y = Wx + b 수행해줌
                logists = self.qg_outputs(sequence_output)
                # softmax 수행
                prob = softmax(logists)
                    
                # argmax 수행
                q_gen = argmax(prob)
                    
                    
                #start_logits, end_logits = logits.split(1,dim=-1)
                #start_logits = start_logits.squeeze(-1)
                #end_logits = end_logits.squeeze(-1)
                    
                #outputs = (start_logits, end_logits,) + outputs[2:]
                if start_positions is not None and end_positions is not None:
                    ignored_index = start_lgist.size(1)
                    start_positions.clamp_(0, ignored_index)
                    end_positions.clamp_(0,ignored_index)
                            
                    loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                    start_loss = loss_fct(start_logits, start_positions)
                    end_loss = loss_fct(end_logits, end_positions)
                    total_loss = (start_loss + end_loss) / 2
                    outputs = (total_loss,) + outputs
                    """
                    dropout_output = self.dropout(last_hidden_states)
                    linear_output = self.linear(last_hidden_states)
                    proba = self.sigmoid(linear_output)
                    """
                return outputs