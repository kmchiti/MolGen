from abc import ABC
from transformers import GPT2Config, GPT2LMHeadModel, AutoConfig
from typing import Optional

class GPT2MolGen(GPT2LMHeadModel, ABC):
    def __init__(
            self,
            model_name_or_path: str,
            max_seq_length: Optional[int],
            vocab_size: Optional[int],
            bos_token_id: Optional[str],
            eos_token_id: Optional[str],
            use_flash_attention_2: Optional[bool] = False,
    ):
        config = AutoConfig.from_pretrained(
                            model_name_or_path,
                            vocab_size=vocab_size,
                            n_ctx=max_seq_length,
                            bos_token_id=bos_token_id,
                            eos_token_id=eos_token_id,
                    )
        # TODO: add flash_atten
        GPT2LMHeadModel.__init__(self, config)

# class GPT2MolGen(pl.LightningModule):
#     def __init__(
#         self,
#         model_name_or_path,
#         max_seq_length,
#         vocab_size,
#         bos_token_id,
#         eos_token_id,
#         lr,
#         weight_decay,
#         max_lr,
#     ):
#         super().__init__()
#
#         self.save_hyperparameters()
#
#         config = AutoConfig.from_pretrained(
#             model_name_or_path,
#             vocab_size=vocab_size,
#             n_ctx=max_seq_length,
#             bos_token_id=bos_token_id,
#             eos_token_id=eos_token_id,
#         )
#         self.net = GPT2LMHeadModel(config=config)
#         self.lr = lr
#         self.weight_decay = weight_decay
#         self.max_lr = max_lr
#
#     def forward(self, x):
#         return self.net(
#             input_ids=x["input_ids"],
#             attention_mask=x["attention_mask"],
#             labels=x["labels"],
#         )
#
#     def training_step(self, batch, batch_idx):
#         loss = self(batch).loss
#         self.log(
#             "train_loss",
#             loss,
#             on_epoch=True,
#             on_step=True,
#             sync_dist=True,
#             prog_bar=True,
#         )
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         loss = self(batch).loss
#         self.log(
#             "valid_loss", loss, on_step=True, on_epoch=True, sync_dist=True
#         )
#
#     def configure_optimizers(self):
#         optimizer = AdamW(
#             self.trainer.model.parameters(),  # type: ignore
#             lr=self.lr,
#             weight_decay=self.weight_decay,
#         )
#         scheduler = OneCycleLR(
#             optimizer,
#             max_lr=self.max_lr,
#             total_steps=self.trainer.estimated_stepping_batches,
#         )
#         return [optimizer], [scheduler]
