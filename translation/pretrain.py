from argparse import ArgumentParser
import torch
import torch.nn as nn
import pytorch_lightning as pl
from model.mbart_model import MBartForConditionalGeneration
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from data_util import PretrainDataModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


class LitModelForPretrain(pl.LightningModule):
    def __init__(self, learning_rate, model_name, total_steps, ofname, tokenizer):
        super().__init__()
        # self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.total_steps = total_steps
        self.model = MBartForConditionalGeneration.from_pretrained(model_name)

        self.model.model.set_test_mode()
        self.loss_func = nn.NLLLoss()
        self.tokenizer = tokenizer
        self.ofname = ofname
    
    def training_step(self, batch, batch_idx):
        out = self.model(input_ids=batch['src_input_ids'], 
                        attention_mask=batch['src_attention_mask'])

        out_B = torch.log(out) * batch['src_attention_mask']
        out_I = torch.log(1 - out * batch['src_attention_mask'])
        label = batch['tgt']

        out = torch.stack((out_I, out_B), dim=-1)
        loss = self.loss_func(out.view(-1, 2), label.view(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        out = self.model(input_ids=batch['input_ids'], 
                        attention_mask=batch['attention_mask'])
        out = out * batch['attention_mask']

        outputs = []
        for i, text in enumerate(batch['text']):
            this_out = out[i]
            this_input_ids = batch['input_ids'][i]
            concat_index = batch['offset_mapping'] #batch not supported    
            
            out_str = ''
            text_words = text.split()
            current_word = 0    
            word_str = ""
            for w_pos, word_idx in enumerate(concat_index):
                temp = torch.zeros(1).type_as(out)
                token_str = ""
                for t_pos, token_idx in enumerate(word_idx):
                    # get score from the last token
                    temp[0] = this_out[token_idx]
                    token_str += self.tokenizer.decode(this_input_ids[token_idx])
                token_str = token_str.strip()
                word_str += token_str
                if word_str != text_words[current_word]:
                    continue
                current_word += 1 
                out_str += " " + word_str
                word_str = ""
                this_score = temp.item()
                if this_score > 0.5:
                    out_str += ' |'
            outputs.append(out_str)
        return outputs
    
    def on_predict_epoch_end(self, results):
        with open(self.ofname, "w") as of:
            outdict = {}
            count = 0
            for batch_result in results:
                for step_result in batch_result:
                    outdict[count] = step_result[0]
                    count += 1
            import json
            json.dump(outdict, of, indent=4)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {'scheduler':get_linear_schedule_with_warmup(optimizer, self.total_steps*0.1, self.total_steps),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

def main(hparams):
    tokenizer = AutoTokenizer.from_pretrained(hparams.model_name, src_lang="en_XX")
    litData = PretrainDataModule(tokenizer, hparams.batch_size)

    total_steps = int(len(litData) * hparams.limit_train_batches / hparams.batch_size) * hparams.epoch
    print("Total steps:", total_steps)

    checkpoint_callback = ModelCheckpoint(
        dirpath='lightning_logs/pretrain',
        filename='{epoch:02d}-{train_loss:.2f}-hrnn-translation'
    )

    if hparams.load_checkpoint:
        model = LitModelForPretrain.load_from_checkpoint(
                                            checkpoint_path='./lightning_logs/pretrain/epoch=02-train_loss=0.02-hrnn-translation.ckpt',
                                            learning_rate=hparams.learning_rate, 
                                            model_name=hparams.model_name, 
                                            total_steps=total_steps, 
                                            ofname=hparams.fname,
                                            tokenizer=tokenizer,
                                            )
    else:
        model = LitModelForPretrain(
                            learning_rate=hparams.learning_rate, 
                            model_name=hparams.model_name, 
                            total_steps=total_steps, 
                            ofname=hparams.fname,
                            tokenizer=tokenizer,
                            )

    trainer = pl.Trainer(accelerator=hparams.accelerator, 
                        devices=hparams.devices, 
                        max_epochs=hparams.epoch,
                        limit_train_batches=hparams.limit_train_batches,
                        callbacks=[checkpoint_callback]
                        )

    if hparams.is_test:
        trainer.predict(model, litData)
    else:
        trainer.fit(model, litData)
        trainer.predict(model, litData)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--devices", default=1)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--epoch", default=3, type=int)
    parser.add_argument("--model_name", default="facebook/mbart-large-50", type=str)
    parser.add_argument("--limit_train_batches", default=1.0, type=float)
    parser.add_argument("--is_test", action="store_true")
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--fname", default="./result_files/model_conll2000_result.json", type=str)
    args = parser.parse_args()
    main(args)