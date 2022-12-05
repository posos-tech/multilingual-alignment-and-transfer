import sys
import wandb
import datasets
import traceback
from typing import Callable, Any, Optional

from multilingual_eval.utils import post_on_slack
from multilingual_eval.tokenization.chinese_segmenter import StanfordSegmenter


def wrap_train(train_fn: Callable[[dict, dict, Optional[StanfordSegmenter]],Any], sweep_config: dict, sweep_id: str, zh_segmenter: Optional[StanfordSegmenter]=None, ts=None):
    datasets.disable_progress_bar()
    def train(config=None):
        with wandb.init():
            config = wandb.config
            run = wandb.run

            partial_config = {
                k: v for k, v in config.items() if k in list(sweep_config["parameters"].keys())
            }

            if ts is not None:
                post_on_slack(
                    f"Starting new run {run.name} (id: {run.id})\n\n"
                    + f"config:\n\n```\n{partial_config}\n```",
                    thread_ts=ts,
                )

            try:
                train_fn(
                    config,
                    sweep_config,
                    zh_segmenter,
                )
            except Exception as e:
                print(traceback.print_exc(), file=sys.stderr)

                post_on_slack(
                    (f"Run from sweep: {sweep_id} failed.\n\n" if ts is None else "")
                    + f"Run: {run.name} (id: {run.id})\n\n"
                    + f"Trace:\n\n```\n{traceback.format_exc()}```",
                    thread_ts=ts,
                )

                raise e

    return train