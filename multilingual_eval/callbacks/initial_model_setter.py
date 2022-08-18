from transformers.trainer_callback import TrainerCallback
from transformers import TrainingArguments, TrainerState, TrainerControl


class InitialModelSetterCallback(TrainerCallback):
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
