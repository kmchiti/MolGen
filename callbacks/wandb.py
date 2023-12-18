from transformers.trainer_callback import TrainerCallback
from transformers.integrations.integration_utils import is_wandb_available, rewrite_logs
from transformers.utils import logging, ENV_VARS_TRUE_VALUES, is_torch_tpu_available
import tempfile
import numbers
from pathlib import Path
import os


logger = logging.get_logger(__name__)

# modified code from: https://github.com/huggingface/transformers/blob/71d47f0ad498b7649f11d3a9cca3cd3585e4341f/src/transformers/integrations/integration_utils.py#L665
class WandbCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that logs metrics, media, model checkpoints to [Weight and Biases](https://www.wandb.com/).
    """

    def __init__(self, entity: str, project: str, name: str, config: dict, tags: list, mode: str= 'online',
                 run_evaluation: bool = False, log_weight_grad_norm: bool = False):
        has_wandb = is_wandb_available()
        if not has_wandb:
            raise RuntimeError("WandbCallback requires wandb to be installed. Run `pip install wandb`.")
        if has_wandb:
            import wandb

            self._wandb = wandb
        self._initialized = False
        self.entity = entity
        self.project = project
        self.name = name
        self.config = config
        self.tags = tags
        self.mode = mode
        self.run_evaluation = run_evaluation
        self.log_weight_grad_norm = log_weight_grad_norm

    def setup(self, args, state, model, **kwargs):
        """
        Setup the optional Weights & Biases (*wandb*) integration.

        One can subclass and override this method to customize the setup if needed. Find more information
        [here](https://docs.wandb.ai/guides/integrations/huggingface). You can also override the following environment
        variables:

        Environment:
        - **WANDB_LOG_MODEL** (`str`, *optional*, defaults to `"false"`):
            Whether to log model and checkpoints during training. Can be `"end"`, `"checkpoint"` or `"false"`. If set
            to `"end"`, the model will be uploaded at the end of training. If set to `"checkpoint"`, the checkpoint
            will be uploaded every `args.save_steps` . If set to `"false"`, the model will not be uploaded. Use along
            with [`~transformers.TrainingArguments.load_best_model_at_end`] to upload best model.

            <Deprecated version="5.0">

            Setting `WANDB_LOG_MODEL` as `bool` will be deprecated in version 5 of 🤗 Transformers.

            </Deprecated>
        - **WANDB_WATCH** (`str`, *optional* defaults to `"false"`):
            Can be `"gradients"`, `"all"`, `"parameters"`, or `"false"`. Set to `"all"` to log gradients and
            parameters.
        - **WANDB_PROJECT** (`str`, *optional*, defaults to `"huggingface"`):
            Set this to a custom string to store results in a different project.
        - **WANDB_DISABLED** (`bool`, *optional*, defaults to `False`):
            Whether to disable wandb entirely. Set `WANDB_DISABLED=true` to disable.
        """
        if self._wandb is None:
            return
        self._initialized = True
        if state.is_world_process_zero:
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            if self._wandb.run is None:
                self._wandb.init(entity=self.entity, project=self.project, name=self.name,
                                 config=self.config, tags=self.tags, mode=self.mode)


            # define default x-axis (for latest wandb versions)
            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("train/global_step")
                self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

            # keep track of model topology and gradients, unsupported on TPU
            _watch_model = os.getenv("WANDB_WATCH", "false")
            if not is_torch_tpu_available() and _watch_model in ("all", "parameters", "gradients"):
                self._wandb.watch(model, log=_watch_model, log_freq=max(100, state.logging_steps))
            self._wandb.run._label(code="transformers_trainer")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self._wandb is None:
            return
        hp_search = state.is_hyper_param_search
        if hp_search:
            self._wandb.finish()
            self._initialized = False
            args.run_name = None
        if not self._initialized:
            self.setup(args, state, model, **kwargs)

    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self._wandb is None:
            return
        if self._log_model in ("end", "checkpoint") and self._initialized and state.is_world_process_zero:
            from transformers import Trainer

            fake_trainer = Trainer(args=args, model=model, tokenizer=tokenizer)
            with tempfile.TemporaryDirectory() as temp_dir:
                fake_trainer.save_model(temp_dir)
                metadata = (
                    {
                        k: v
                        for k, v in dict(self._wandb.summary).items()
                        if isinstance(v, numbers.Number) and not k.startswith("_")
                    }
                    if not args.load_best_model_at_end
                    else {
                        f"eval/{args.metric_for_best_model}": state.best_metric,
                        "train/total_floss": state.total_flos,
                    }
                )
                logger.info("Logging model artifacts. ...")
                model_name = (
                    f"model-{self._wandb.run.id}"
                    if (args.run_name is None or args.run_name == args.output_dir)
                    else f"model-{self._wandb.run.name}"
                )
                artifact = self._wandb.Artifact(name=model_name, type="model", metadata=metadata)
                for f in Path(temp_dir).glob("*"):
                    if f.is_file():
                        with artifact.new_file(f.name, mode="wb") as fa:
                            fa.write(f.read_bytes())
                self._wandb.run.log_artifact(artifact)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            logs = rewrite_logs(logs)
            self._wandb.log({**logs, "train/global_step": state.global_step})

    def on_save(self, args, state, control, **kwargs):
        if self._log_model == "checkpoint" and self._initialized and state.is_world_process_zero:
            checkpoint_metadata = {
                k: v
                for k, v in dict(self._wandb.summary).items()
                if isinstance(v, numbers.Number) and not k.startswith("_")
            }

            ckpt_dir = f"checkpoint-{state.global_step}"
            artifact_path = os.path.join(args.output_dir, ckpt_dir)
            logger.info(f"Logging checkpoint artifacts in {ckpt_dir}. ...")
            checkpoint_name = (
                f"checkpoint-{self._wandb.run.id}"
                if (args.run_name is None or args.run_name == args.output_dir)
                else f"checkpoint-{self._wandb.run.name}"
            )
            artifact = self._wandb.Artifact(name=checkpoint_name, type="model", metadata=checkpoint_metadata)
            artifact.add_dir(artifact_path)
            self._wandb.log_artifact(artifact, aliases=[f"checkpoint-{state.global_step}"])

    def on_evaluate(self, args, state, control, **kwargs):
        """
        Event called after an evaluation phase.
        """
        pass

    def on_substep_end(self, args, state, control, model=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            logs = get_weight_grad_norm(model)
            self._wandb.log({**logs, "train/global_step": state.global_step})


def get_weight_grad_norm(model):
    weight_norm, grad_norm = 0, 0
    for param in model.parameters():
        grad_norm += param.grad.data.norm(2).item() ** 2 if param.grad is not None else 0
        weight_norm += param.data.norm(2).item() ** 2
    res = {"model/weight_norm": weight_norm**0.5, "model/grad_norm": grad_norm**0.5}
    return res


# from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
# import numpy as np
# class EvalLoopOutput(NamedTuple):
#     predictions: Union[np.ndarray, Tuple[np.ndarray]]
#     label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
#     metrics: Optional[Dict[str, float]]
#     num_samples: Optional[int]
#
#  # Metrics!
#         if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
#             if args.include_inputs_for_metrics:
#                 metrics = self.compute_metrics(
#                     EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
#                 )
#             else:
#                 metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
#         else:
#             metrics = {}
#
#         # To be JSON-serializable, we need to remove numpy types or zero-d tensors
#         metrics = denumpify_detensorize(metrics)