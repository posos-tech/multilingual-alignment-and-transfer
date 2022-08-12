from typing import List, Optional
from transformers import Trainer
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertForTokenClassification,
    BertForSequenceClassification,
)
import torch
import torch.nn.functional as F
from torch.nn import DataParallel


def compute_realignment_loss(
    encoder,
    realignment_transformation,
    realignment_layers: List[int],
    strong_alignment=False,
    realignment_temperature=0.1,
    realignment_coef=1.0,
    alignment_hidden_size=128,
    # sentences from left language
    left_input_ids=None,
    left_attention_mask=None,
    left_token_type_ids=None,
    left_position_ids=None,
    left_head_mask=None,
    left_inputs_embeds=None,
    # sentences from right language
    right_input_ids=None,
    right_attention_mask=None,
    right_token_type_ids=None,
    right_position_ids=None,
    right_head_mask=None,
    right_inputs_embeds=None,
    # alignment labels
    alignment_left_ids=None,  # [word_id] -> [[word_id | -1]]
    alignment_left_positions=None,  # [[batch, start, end]] -> [[[start, end]]]
    alignment_right_ids=None,  # [word_id]
    alignment_right_positions=None,  # [[batch, start, end]]
    alignment_nb=None,  # [[i]]
    alignment_left_length=None,  # [[i]]
    alignment_right_length=None,  # [[i]]
):
    total_loss = None
    left_output = encoder(
        left_input_ids,
        attention_mask=left_attention_mask,
        token_type_ids=left_token_type_ids,
        position_ids=left_position_ids,
        head_mask=left_head_mask,
        inputs_embeds=left_inputs_embeds,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
    )

    left_hidden_states = left_output.hidden_states

    right_output = encoder(
        right_input_ids,
        attention_mask=right_attention_mask,
        token_type_ids=right_token_type_ids,
        position_ids=right_position_ids,
        head_mask=right_head_mask,
        inputs_embeds=right_inputs_embeds,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
    )

    right_hidden_states = right_output.hidden_states

    the_device = left_hidden_states[0].device

    for layer in realignment_layers:
        # Inspired by https://github.com/shijie-wu/crosslingual-nlp/blob/780f738df2b75f653aaaf11b9f513850fe11ba36/src/model/aligner.py#L139

        aligned_left_repr = torch.zeros(
            (alignment_left_ids.shape[0], alignment_left_ids.shape[1], alignment_hidden_size),
            device=the_device,
        )
        aligned_right_repr = torch.zeros(
            (alignment_left_ids.shape[0], alignment_left_ids.shape[1], alignment_hidden_size),
            device=the_device,
        )

        for b in range(alignment_left_ids.shape[0]):
            for i in range(alignment_left_ids.shape[1]):
                aligned_left_repr[b, i] = realignment_transformation(
                    torch.sum(
                        left_hidden_states[layer][
                            b,
                            alignment_left_positions[b, alignment_left_ids[b, i]][
                                0
                            ] : alignment_left_positions[b, alignment_left_ids[b, i]][1],
                        ],
                        0,
                    )
                )
                aligned_right_repr[b, i] = realignment_transformation(
                    torch.sum(
                        right_hidden_states[layer][
                            b,
                            alignment_right_positions[b, alignment_right_ids[b, i]][
                                0
                            ] : alignment_right_positions[b, alignment_right_ids[b, i]][1],
                        ],
                        0,
                    )
                )

        all_left_repr = torch.zeros(
            (
                alignment_left_positions.shape[0],
                alignment_left_positions.shape[1],
                alignment_hidden_size,
            ),
            device=the_device,
        )
        all_right_repr = torch.zeros(
            (
                alignment_right_positions.shape[0],
                alignment_right_positions.shape[1],
                alignment_hidden_size,
            ),
            device=the_device,
        )
        for b in range(alignment_right_positions.shape[0]):
            for i in range(alignment_left_positions.shape[1]):
                all_left_repr[b, i] = realignment_transformation(
                    torch.sum(
                        left_hidden_states[layer][
                            b,
                            alignment_left_positions[b, i][0] : alignment_left_positions[b, i][1],
                        ],
                        0,
                    )
                )
            for i in range(alignment_right_positions.shape[1]):
                all_right_repr[b, i] = realignment_transformation(
                    torch.sum(
                        right_hidden_states[layer][
                            b,
                            alignment_right_positions[b, i][0] : alignment_right_positions[b, i][1],
                        ],
                        0,
                    )
                )

        aligned_left_repr = torch.cat(
            (*[aligned_left_repr[b][: alignment_nb[b]] for b in range(aligned_left_repr.shape[0])],)
        )
        aligned_right_repr = torch.cat(
            (
                *[
                    aligned_right_repr[b][: alignment_nb[b]]
                    for b in range(aligned_right_repr.shape[0])
                ],
            )
        )
        all_left_repr = torch.cat(
            (
                *[
                    all_left_repr[b][: alignment_left_length[b]]
                    for b in range(all_left_repr.shape[0])
                ],
            )
        )
        all_right_repr = torch.cat(
            (
                *[
                    all_right_repr[b][: alignment_right_length[b]]
                    for b in range(all_right_repr.shape[0])
                ],
            )
        )

        right_cumul_length = torch.cat(
            (
                torch.tensor([0], dtype=torch.long, device=the_device),
                torch.cumsum(alignment_right_length, 0),
            )
        )
        left_cumul_length = torch.cat(
            (
                torch.tensor([0], dtype=torch.long, device=the_device),
                torch.cumsum(alignment_left_length, 0),
            )
        )

        left_goal = torch.cat(
            (
                *[
                    all_left_repr.shape[0]
                    + right_cumul_length[b]
                    + alignment_right_ids[b][: alignment_nb[b]]
                    for b in range(alignment_left_ids.shape[0])
                ],
            )
        )
        right_goal = torch.cat(
            (
                *[
                    left_cumul_length[b] + alignment_left_ids[b][: alignment_nb[b]]
                    for b in range(alignment_right_ids.shape[0])
                ],
            )
        )

        aligned_reprs = torch.cat((aligned_left_repr, aligned_right_repr))
        all_reprs = torch.cat((all_left_repr, all_right_repr))
        sim = torch.matmul(aligned_reprs, all_reprs.transpose(0, 1))
        aligned_norms = aligned_reprs.norm(dim=1, keepdim=True)
        all_norms = all_reprs.norm(dim=1, keepdim=True)

        sim /= aligned_norms
        sim /= all_norms.transpose(0, 1)
        sim /= realignment_temperature

        if not strong_alignment:
            # remove same-language similarities
            sim[: aligned_left_repr.shape[0], : all_left_repr.shape[0]] -= 1e6
            sim[aligned_left_repr.shape[0] :, all_left_repr.shape[0] :] -= 1e6
        else:
            # remove (x,x) similarities
            sim[
                torch.arange(0, aligned_left_repr.shape[0], 1, device=the_device), right_goal,
            ] -= 1e6
            sim[
                torch.arange(
                    aligned_right_repr.shape[0],
                    2 * aligned_right_repr.shape[0],
                    1,
                    device=the_device,
                ),
                left_goal,
            ] -= 1e6

        logits = F.log_softmax(sim, dim=-1)
        goal = torch.cat((left_goal, right_goal))

        loss = F.nll_loss(logits, goal)

        if total_loss is None:
            total_loss = (realignment_coef / len(realignment_layers)) * loss
        else:
            total_loss += (realignment_coef / len(realignment_layers)) * loss

    return total_loss


class BertForTokenClassificationWithRealignment(BertForTokenClassification):
    def __init__(
        self,
        config,
        realignment_layers: Optional[List[int]] = None,
        realignment_transformation: Optional[List[int]] = None,
        strong_alignment=False,
        realignment_temperature=0.1,
        realignment_coef=1.0,
    ):
        super(BertForTokenClassificationWithRealignment, self).__init__(config)

        realignment_layers = realignment_layers or [-1]
        if isinstance(realignment_layers, int):
            realignment_layers = [realignment_layers]
        n_layers = config.num_hidden_layers
        realignment_layers = list(map(lambda x: x if x >= 0 else n_layers + x, realignment_layers))
        self.realignment_layers = realignment_layers

        if realignment_transformation is None:
            realignment_transformation = [config.hidden_size, 128]
        transformations = []
        for i, v in enumerate(realignment_transformation):
            if i == 0:
                transformations.append(torch.nn.Linear(config.hidden_size, v, bias=False))
            else:
                transformations.append(
                    torch.nn.Linear(realignment_transformation[i - 1], v, bias=False)
                )

            if i < len(realignment_transformation) - 1:
                transformations.append(torch.nn.ReLU())
        if len(transformations) > 0:
            self.realignment_transformation = torch.nn.Sequential(*transformations)
        else:
            self.realignment_transformation = lambda x: x
        self.alignment_hidden_size = (
            realignment_transformation[-1]
            if len(realignment_transformation) > 0
            else config.hidden_size
        )

        self.strong_alignment = strong_alignment

        self.realignment_temperature = realignment_temperature

        self.realignment_coef = realignment_coef

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
        # sentences from left language
        left_input_ids=None,
        left_attention_mask=None,
        left_token_type_ids=None,
        left_position_ids=None,
        left_head_mask=None,
        left_inputs_embeds=None,
        # sentences from right language
        right_input_ids=None,
        right_attention_mask=None,
        right_token_type_ids=None,
        right_position_ids=None,
        right_head_mask=None,
        right_inputs_embeds=None,
        # alignment labels
        alignment_left_ids=None,
        alignment_left_positions=None,
        alignment_right_ids=None,
        alignment_right_positions=None,
        alignment_nb=None,
        alignment_left_length=None,
        alignment_right_length=None,
    ):
        if input_ids is not None and left_input_ids is not None:
            res = super(BertForTokenClassificationWithRealignment, self).forward(
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                inputs_embeds,
                labels,
                output_attentions,
                output_hidden_states,
                return_dict=return_dict,
            )
            if labels is not None:
                realignment_loss = compute_realignment_loss(
                    self.bert,
                    self.realignment_transformation,
                    self.realignment_layers,
                    self.strong_alignment,
                    self.realignment_temperature,
                    self.realignment_coef,
                    self.alignment_hidden_size,
                    left_input_ids,
                    left_attention_mask,
                    left_token_type_ids,
                    left_position_ids,
                    left_head_mask,
                    left_inputs_embeds,
                    right_input_ids,
                    right_attention_mask,
                    right_token_type_ids,
                    right_position_ids,
                    right_head_mask,
                    right_inputs_embeds,
                    alignment_left_ids,
                    alignment_left_positions,
                    alignment_right_ids,
                    alignment_right_positions,
                    alignment_nb,
                    alignment_left_length,
                    alignment_right_length,
                )
                if return_dict:
                    res.loss += realignment_loss
                    return res
                else:
                    return (res[0] + realignment_loss, *res[1:])
        if input_ids is not None:
            return super(BertForTokenClassificationWithRealignment, self).forward(
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                inputs_embeds,
                labels,
                output_attentions,
                output_hidden_states,
                return_dict=return_dict,
            )
        else:
            realignment_loss = compute_realignment_loss(
                self.bert,
                self.realignment_transformation,
                self.realignment_layers,
                self.strong_alignment,
                self.realignment_temperature,
                self.realignment_coef,
                self.alignment_hidden_size,
                left_input_ids,
                left_attention_mask,
                left_token_type_ids,
                left_position_ids,
                left_head_mask,
                left_inputs_embeds,
                right_input_ids,
                right_attention_mask,
                right_token_type_ids,
                right_position_ids,
                right_head_mask,
                right_inputs_embeds,
                alignment_left_ids,
                alignment_left_positions,
                alignment_right_ids,
                alignment_right_positions,
                alignment_nb,
                alignment_left_length,
                alignment_right_length,
            )
            if not return_dict:
                return (realignment_loss,)
            return TokenClassifierOutput(loss=realignment_loss)

    def compute_additional_loss(
        self,
        # sentences from left language
        left_input_ids=None,
        left_attention_mask=None,
        left_token_type_ids=None,
        left_position_ids=None,
        left_head_mask=None,
        left_inputs_embeds=None,
        # sentences from right language
        right_input_ids=None,
        right_attention_mask=None,
        right_token_type_ids=None,
        right_position_ids=None,
        right_head_mask=None,
        right_inputs_embeds=None,
        # alignment labels
        alignment_left_ids=None,
        alignment_left_positions=None,
        alignment_right_ids=None,
        alignment_right_positions=None,
        alignment_nb=None,
        alignment_left_length=None,
        alignment_right_length=None,
    ):
        return compute_realignment_loss(
            self.bert,
            self.realignment_transformation,
            self.realignment_layers,
            self.strong_alignment,
            self.realignment_temperature,
            self.realignment_coef,
            self.alignment_hidden_size,
            left_input_ids,
            left_attention_mask,
            left_token_type_ids,
            left_position_ids,
            left_head_mask,
            left_inputs_embeds,
            right_input_ids,
            right_attention_mask,
            right_token_type_ids,
            right_position_ids,
            right_head_mask,
            right_inputs_embeds,
            alignment_left_ids,
            alignment_left_positions,
            alignment_right_ids,
            alignment_right_positions,
            alignment_nb,
            alignment_left_length,
            alignment_right_length,
        )


class BertForSequenceClassificationWithRealignment(BertForSequenceClassification):
    def __init__(
        self,
        config,
        realignment_layers: Optional[List[int]] = None,
        realignment_transformation: Optional[List[int]] = None,
        strong_alignment=False,
        realignment_temperature=0.1,
        realignment_coef=1.0,
    ):
        super(BertForSequenceClassificationWithRealignment, self).__init__(config)

        realignment_layers = realignment_layers or [-1]
        if isinstance(realignment_layers, int):
            realignment_layers = [realignment_layers]
        n_layers = config.num_hidden_layers
        realignment_layers = list(map(lambda x: x if x >= 0 else n_layers + x, realignment_layers))
        self.realignment_layers = realignment_layers

        if realignment_transformation is None:
            realignment_transformation = [config.hidden_size, 128]
        transformations = []
        for i, v in enumerate(realignment_transformation):
            if i == 0:
                transformations.append(torch.nn.Linear(config.hidden_size, v, bias=False))
            else:
                transformations.append(
                    torch.nn.Linear(realignment_transformation[i - 1], v, bias=False)
                )

            if i < len(realignment_transformation) - 1:
                transformations.append(torch.nn.ReLU())
        if len(transformations) > 0:
            self.realignment_transformation = torch.nn.Sequential(*transformations)
        else:
            self.realignment_transformation = lambda x: x
        self.alignment_hidden_size = (
            realignment_transformation[-1]
            if len(realignment_transformation) > 0
            else config.hidden_size
        )

        self.strong_alignment = strong_alignment

        self.realignment_temperature = realignment_temperature

        self.realignment_coef = realignment_coef

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
        # sentences from left language
        left_input_ids=None,
        left_attention_mask=None,
        left_token_type_ids=None,
        left_position_ids=None,
        left_head_mask=None,
        left_inputs_embeds=None,
        # sentences from right language
        right_input_ids=None,
        right_attention_mask=None,
        right_token_type_ids=None,
        right_position_ids=None,
        right_head_mask=None,
        right_inputs_embeds=None,
        # alignment labels
        alignment_left_ids=None,
        alignment_left_positions=None,
        alignment_right_ids=None,
        alignment_right_positions=None,
        alignment_nb=None,
        alignment_left_length=None,
        alignment_right_length=None,
    ):
        if input_ids is not None and left_input_ids is not None:
            raise Exception(
                f"{self.__name__} was given a batch containing simultaneously normal and realignment inputs. A batch should contain either one of them but not both."
            )
        if input_ids is not None:
            return super(BertForSequenceClassificationWithRealignment, self).forward(
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                inputs_embeds,
                labels,
                output_attentions,
                output_hidden_states,
                return_dict=True,
            )
        else:
            realignment_loss = compute_realignment_loss(
                self.bert,
                self.realignment_transformation,
                self.realignment_layers,
                self.strong_alignment,
                self.realignment_temperature,
                self.realignment_coef,
                self.alignment_hidden_size,
                left_input_ids,
                left_attention_mask,
                left_token_type_ids,
                left_position_ids,
                left_head_mask,
                left_inputs_embeds,
                right_input_ids,
                right_attention_mask,
                right_token_type_ids,
                right_position_ids,
                right_head_mask,
                right_inputs_embeds,
                alignment_left_ids,
                alignment_left_positions,
                alignment_right_ids,
                alignment_right_positions,
                alignment_nb,
                alignment_left_length,
                alignment_right_length,
            )
            if not return_dict:
                return (realignment_loss,)
            return TokenClassifierOutput(loss=realignment_loss)

    def compute_additional_loss(
        self,
        # sentences from left language
        left_input_ids=None,
        left_attention_mask=None,
        left_token_type_ids=None,
        left_position_ids=None,
        left_head_mask=None,
        left_inputs_embeds=None,
        # sentences from right language
        right_input_ids=None,
        right_attention_mask=None,
        right_token_type_ids=None,
        right_position_ids=None,
        right_head_mask=None,
        right_inputs_embeds=None,
        # alignment labels
        alignment_left_ids=None,
        alignment_left_positions=None,
        alignment_right_ids=None,
        alignment_right_positions=None,
        alignment_nb=None,
        alignment_left_length=None,
        alignment_right_length=None,
    ):
        return compute_realignment_loss(
            self.bert,
            self.realignment_transformation,
            self.realignment_layers,
            self.strong_alignment,
            self.realignment_temperature,
            self.realignment_coef,
            self.alignment_hidden_size,
            left_input_ids,
            left_attention_mask,
            left_token_type_ids,
            left_position_ids,
            left_head_mask,
            left_inputs_embeds,
            right_input_ids,
            right_attention_mask,
            right_token_type_ids,
            right_position_ids,
            right_head_mask,
            right_inputs_embeds,
            alignment_left_ids,
            alignment_left_positions,
            alignment_right_ids,
            alignment_right_positions,
            alignment_nb,
            alignment_left_length,
            alignment_right_length,
        )


class TrainerWithRealignment(Trainer):
    def __init__(
        self, *args, realignment_dataloader=None, **kwargs,
    ):
        super(TrainerWithRealignment, self).__init__(*args, **kwargs)

        self.realignment_dataloader = realignment_dataloader
        if realignment_dataloader is not None:
            self.realignment_iterator = iter(self.realignment_dataloader)

    def compute_loss(self, model, inputs, return_outputs=False):
        res = super(TrainerWithRealignment, self).compute_loss(
            model, inputs, return_outputs=return_outputs
        )

        if self.realignment_dataloader is None:
            return res

        if return_outputs:
            loss, outputs = res
        else:
            loss = res

        try:
            realignment_batch = next(self.realignment_iterator)
        except StopIteration:
            self.realignment_iterator = iter(self.realignment_dataloader)
            realignment_batch = next(self.realignment_iterator)
        realignment_batch = self._prepare_inputs(realignment_batch)

        if isinstance(model, DataParallel):
            additional_loss = model.module.compute_additional_loss(**realignment_batch)
        else:
            additional_loss = model.compute_additional_loss(**realignment_batch)

        loss += additional_loss

        return (loss, outputs) if return_outputs else loss
