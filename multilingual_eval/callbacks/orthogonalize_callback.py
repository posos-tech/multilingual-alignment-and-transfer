from typing import Union
from transformers.trainer_callback import TrainerCallback
from transformers import TrainingArguments, TrainerState, TrainerControl

from multilingual_eval.models.with_realignment_factory import (
    BertForTokenClassificationWithRealignment,
    BertForSequenceClassificationWithRealignment,
)


class OrthogonalizeMappingCallback(TrainerCallback):
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Union[
            BertForTokenClassificationWithRealignment, BertForSequenceClassificationWithRealignment
        ] = None,
        **kwargs
    ):
        if model.bert.mapping is not None:
            model.bert.mapping.orthogonalize()
