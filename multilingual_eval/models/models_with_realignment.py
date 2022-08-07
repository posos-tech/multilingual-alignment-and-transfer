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
    alignment_left_ids=None,  # [word_id]
    alignment_left_positions=None,  # [[batch, start, end]]
    alignment_right_ids=None,  # [word_id]
    alignment_right_positions=None,  # [[batch, start, end]]
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
            (alignment_left_ids.shape[0], alignment_hidden_size), device=the_device,
        )
        aligned_right_repr = torch.zeros(
            (alignment_left_ids.shape[0], alignment_hidden_size), device=the_device,
        )

        for i in range(alignment_left_ids.shape[0]):
            aligned_left_repr[i] = realignment_transformation(
                torch.sum(
                    left_hidden_states[layer][
                        alignment_left_positions[alignment_left_ids[i]][0],
                        alignment_left_positions[alignment_left_ids[i]][
                            1
                        ] : alignment_left_positions[alignment_left_ids[i]][2],
                    ],
                    0,
                )
            )
            aligned_right_repr[i] = realignment_transformation(
                torch.sum(
                    right_hidden_states[layer][
                        alignment_right_positions[alignment_right_ids[i]][0],
                        alignment_right_positions[alignment_right_ids[i]][
                            1
                        ] : alignment_right_positions[alignment_right_ids[i]][2],
                    ],
                    0,
                )
            )

        all_left_repr = torch.zeros(
            (alignment_left_positions.shape[0], alignment_hidden_size), device=the_device,
        )
        all_right_repr = torch.zeros(
            (alignment_right_positions.shape[0], alignment_hidden_size), device=the_device,
        )
        for i in range(alignment_left_positions.shape[0]):
            all_left_repr[i] = realignment_transformation(
                torch.sum(
                    left_hidden_states[layer][
                        alignment_left_positions[i][0],
                        alignment_left_positions[i][1] : alignment_left_positions[i][2],
                    ],
                    0,
                )
            )
        for i in range(alignment_right_positions.shape[0]):
            all_right_repr[i] = realignment_transformation(
                torch.sum(
                    right_hidden_states[layer][
                        alignment_right_positions[i][0],
                        alignment_right_positions[i][1] : alignment_right_positions[i][2],
                    ],
                    0,
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
                torch.arange(0, aligned_left_repr.shape[0], 1, device=the_device),
                alignment_left_ids,
            ] -= 1e6
            sim[
                torch.arange(
                    aligned_right_repr.shape[0],
                    2 * aligned_right_repr.shape[0],
                    1,
                    device=the_device,
                ),
                alignment_right_ids + all_left_repr.shape[0],
            ] -= 1e6

        logits = F.log_softmax(sim, dim=-1)

        loss = F.nll_loss(
            logits, torch.cat((alignment_right_ids + all_left_repr.shape[0], alignment_left_ids))
        )

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
        self.realignment_transformation = torch.nn.Sequential(*transformations)
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
        alignment_left_ids=None,  # [word_id]
        alignment_left_positions=None,  # [[batch, start, end]]
        alignment_right_ids=None,  # [word_id]
        alignment_right_positions=None,  # [[batch, start, end]]
    ):
        if input_ids is not None and left_input_ids is not None:
            raise Exception(
                f"{self.__name__} was given a batch containing simultaneously normal and realignment inputs. A batch should contain either one of them but not both."
            )
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
                alignment_left_ids,  # [word_id]
                alignment_left_positions,  # [[batch, start, end]]
                alignment_right_ids,  # [word_id]
                alignment_right_positions,  # [[batch, start, end]]
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
        alignment_left_ids=None,  # [word_id]
        alignment_left_positions=None,  # [[batch, start, end]]
        alignment_right_ids=None,  # [word_id]
        alignment_right_positions=None,  # [[batch, start, end]]
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
            alignment_left_ids,  # [word_id]
            alignment_left_positions,  # [[batch, start, end]]
            alignment_right_ids,  # [word_id]
            alignment_right_positions,  # [[batch, start, end]]
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
        self.realignment_transformation = torch.nn.Sequential(*transformations)
        self.alignment_hidden_size = (
            realignment_transformation[-1]
            if len(realignment_transformation) > 0
            else config.hidden_size
        )

        self.strong_alignment = strong_alignment

        self.realignment_temperature = realignment_temperature

        self.realignment_coef = realignment_coef

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
        alignment_left_ids=None,  # [word_id]
        alignment_left_positions=None,  # [[batch, start, end]]
        alignment_right_ids=None,  # [word_id]
        alignment_right_positions=None,  # [[batch, start, end]]
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
            alignment_left_ids,  # [word_id]
            alignment_left_positions,  # [[batch, start, end]]
            alignment_right_ids,  # [word_id]
            alignment_right_positions,  # [[batch, start, end]]
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


if __name__ == "__main__":

    from datasets import load_dataset
    from transformers import AutoTokenizer, BertForTokenClassification, DataCollatorWithPadding
    import argparse
    import sys, os

    sys.path.append(os.curdir)

    from multilingual_eval.datasets.realignment_dataset import DatasetMapperForRealignment

    parser = argparse.ArgumentParser()
    parser.add_argument("dico_path", type=str)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = BertForTokenClassificationWithRealignment.from_pretrained(
        "bert-base-multilingual-cased", num_labels=17
    )

    translation_dataset = load_dataset("news_commentary", "de-en")

    def preprocess_news_commentary(example):
        return {k: v for k, v in example["translation"].items()}

    translation_dataset = translation_dataset.map(
        preprocess_news_commentary, remove_columns=["id", "translation"]
    ).filter(lambda x: x["en"] is not None and x["de"] is not None)

    dataset_transformer = DatasetMapperForRealignment(tokenizer, args.dico_path, "en", "de")

    # for i, elt in enumerate(translation_dataset["train"]):
    #     if i == 5:
    #         break
    #     dataset_transformer(elt)

    # print(dataset_transformer.perfs)

    realignment_dataset = translation_dataset.map(dataset_transformer, remove_columns=["en", "de"])
    realignment_dataset = realignment_dataset.filter(lambda x: len(x["alignment_left_ids"]) > 0)

    alignment_iterator = iter(realignment_dataset["train"])

    first_realignment_example = next(alignment_iterator)

    print(first_realignment_example)

    second_realignment_example = next(alignment_iterator)

    print(second_realignment_example)

    usual_collator = DataCollatorWithPadding(tokenizer)

    def small_collator_fn(examples):
        left_inputs = [
            {k.split("_", 1)[1]: v for k, v in sample.items() if k.startswith("left_")}
            for sample in examples
        ]
        right_inputs = [
            {k.split("_", 1)[1]: v for k, v in sample.items() if k.startswith("right_")}
            for sample in examples
        ]

        batch_left = {f"left_{k}": v for k, v in usual_collator(left_inputs).items()}
        batch_right = {f"right_{k}": v for k, v in usual_collator(right_inputs).items()}

        alignment_left_ids = None
        alignment_right_ids = None
        alignment_left_positions = None
        alignment_right_positions = None
        offset_left = 0
        offset_right = 0
        for i, example in enumerate(examples):
            sample_alignment_left_ids = example["alignment_left_ids"]
            sample_alignment_right_ids = example["alignment_right_ids"]
            sample_alignment_left_positions = example["alignment_left_positions"]
            sample_alignment_right_positions = example["alignment_right_positions"]

            sample_alignment_left_ids = torch.tensor(sample_alignment_left_ids, dtype=torch.long)
            sample_alignment_right_ids = torch.tensor(sample_alignment_right_ids, dtype=torch.long)
            sample_alignment_left_positions = torch.tensor(
                sample_alignment_left_positions, dtype=torch.long
            )
            sample_alignment_right_positions = torch.tensor(
                sample_alignment_right_positions, dtype=torch.long
            )

            sample_alignment_left_ids += offset_left
            sample_alignment_right_ids += offset_right

            sample_alignment_left_positions[:, 0] = i
            sample_alignment_right_positions[:, 0] = i

            offset_left += sample_alignment_left_positions.shape[0]
            offset_right += sample_alignment_right_positions.shape[0]

            alignment_left_ids = (
                sample_alignment_left_ids
                if alignment_left_ids is None
                else torch.cat((alignment_left_ids, sample_alignment_left_ids))
            )
            alignment_right_ids = (
                sample_alignment_right_ids
                if alignment_right_ids is None
                else torch.cat((alignment_right_ids, sample_alignment_right_ids))
            )
            alignment_left_positions = (
                sample_alignment_left_positions
                if alignment_left_positions is None
                else torch.cat((alignment_left_positions, sample_alignment_left_positions))
            )
            alignment_right_positions = (
                sample_alignment_right_positions
                if alignment_right_positions is None
                else torch.cat((alignment_right_positions, sample_alignment_right_positions))
            )

        return {
            **batch_left,
            **batch_right,
            "alignment_left_ids": alignment_left_ids,
            "alignment_right_ids": alignment_right_ids,
            "alignment_left_positions": alignment_left_positions,
            "alignment_right_positions": alignment_right_positions,
        }

    batch = small_collator_fn([first_realignment_example, second_realignment_example])

    loss = compute_realignment_loss(model, lambda x: x, [-1], **batch)

    print(loss)

