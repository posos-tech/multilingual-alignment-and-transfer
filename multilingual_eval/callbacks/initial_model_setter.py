from transformers.trainer_callback import TrainerCallback
from transformers import TrainingArguments, TrainerState, TrainerControl


class InitialModelSetterCallback(TrainerCallback):
    """
    Callback for HF Trainer for storing a copy of the model at a given point of the training
    in order to perform some kind of regularization afterwards as in Cao et al. 2020
    where l2 loss is used to avoid the representations built by the model to diverge too much from
    those of the original one.

    This only work with models built with multilingual_eval.models.with_realignment_factory.model_with_realignment_factory

    http://nlp.cs.berkeley.edu/pubs/Cao-Kitaev-Klein_2020_MultilingualAlignment_paper.pdf
    """

    def __init__(self, step: int):
        self.step = step
        self.took_effect = False

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        if state.global_step < self.step:
            return
        if state.global_step >= self.step and not self.took_effect:
            model.reset_initial_weights_regularizer()
