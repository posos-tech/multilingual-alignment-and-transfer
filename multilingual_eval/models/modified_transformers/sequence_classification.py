from typing import Optional, Union, Tuple
import torch
from transformers import BertForSequenceClassification, RobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from multilingual_eval.models.modified_transformers.bert_model import (
    BertModelWithOptionalMapping,
    encoder_with_optional_mapping_factory,
)


def sequence_classifier_with_optional_mapping_factory(BaseClass):
    """
    Factory function for creating a custom class for a token classification model
    that builds on top of an existing one (BaseClass) by adding an optional orthogonal
    mapping to the encoder (and lang_id in the forward method) 
    """

    class CustomModelForSequenceClassification(BaseClass):
        """
        Custom class for a token classification model
        that builds on top of an existing one (BaseClass) by adding an optional orthogonal
        mapping to the encoder (and lang_id in the forward method) 
        """
        def __init__(self, config, with_mapping=False, nb_pairs=1):
            super().__init__(config)

            encoder_class = getattr(self, BaseClass.base_model_prefix).__class__

            setattr(
                self,
                BaseClass.base_model_prefix,
                encoder_with_optional_mapping_factory(encoder_class)(
                    config, add_pooling_layer=False, with_mapping=with_mapping, nb_pairs=nb_pairs
                ),
            )

            self.post_init()

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            lang_id: Optional[torch.Tensor] = None,
            train_only_mapping=False,
        ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
            r"""
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            """
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = getattr(self, BaseClass.base_model_prefix)(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                lang_id=lang_id,  # Had to rewrite the forward method because of this line
                train_only_mapping=train_only_mapping,
            )

            pooled_output = outputs[1]

            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (
                        labels.dtype == torch.long or labels.dtype == torch.int
                    ):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = torch.nn.MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = torch.nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)
            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    return CustomModelForSequenceClassification


CustomBertForSequenceClassification = sequence_classifier_with_optional_mapping_factory(
    BertForSequenceClassification,
)
CustomRobertaForSequenceClassification = sequence_classifier_with_optional_mapping_factory(
    RobertaForSequenceClassification,
)
