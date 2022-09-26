from typing import Union
from transformers.trainer_callback import TrainerCallback
from transformers import TrainingArguments, TrainerState, TrainerControl

from multilingual_eval.models.with_realignment_factory import (
    BertForTokenClassificationWithRealignment,
    BertForSequenceClassificationWithRealignment,
)


class OrthogonalizeMappingCallback(TrainerCallback):
    """
    Callback for HF Trainer which perform a regularization step that enforce the 'mapping'
    attribute of the trained model to by an orthogonal matrix

    This only work with models built with multilingual_eval.models.with_realignment_factory.model_with_realignment_factory
    AND having a BERT encoder (does not work with RoBERTa)
    """

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
