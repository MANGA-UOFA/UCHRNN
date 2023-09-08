from argparse import ArgumentParser
import torch
import torch.nn as nn
import pytorch_lightning as pl
from model.mbart_model import MBartForConditionalGeneration
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from data_util import TranslationDataModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from rouge import Rouge
import math

class Hparams_hrnn:
    def __init__(self, auxloss_coefficient, layers_ratio, attn_head_mask_ratio, reweight_coefficient):
        self.auxloss_coefficient = auxloss_coefficient
        self.layers_ratio = layers_ratio
        self.attn_head_mask_ratio = attn_head_mask_ratio
        self.reweight_coefficient = reweight_coefficient
        # self.layers=[1,3,5,7,9,11]

class LitModel(pl.LightningModule):
    def __init__(self, learning_rate, model_name, total_steps, hparams_hrnn, tokenizer):
        super().__init__()
        # self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.total_steps = total_steps
        self.model = MBartForConditionalGeneration.from_pretrained(model_name)
        self.model.set_summarization_hparams(hparams_hrnn)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="en_XX", tgt_lang="de_DE")
        self.tokenizer = tokenizer
        self.chunk_cutoff = 0.5
        self.this_step = 0
        self.fname = None

    def training_step(self, batch, batch_idx):
        # discout_rate = 1 - self.this_step / self.total_steps
        discout_rate = 1
        out = self.model(input_ids=batch['src_input_ids'], 
                        attention_mask=batch['src_attention_mask'], 
                        labels=batch['tgt_input_ids'],
                        decoder_attention_mask=batch['tgt_attention_mask'], # should be added to cedar later
                        mse_label=batch['mse_label'],
                        discout_rate=discout_rate)
        loss = out.loss
        self.this_step += 1
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        out = self.model.generate(input_ids=batch['src_input_ids'], forced_bos_token_id=self.tokenizer.lang_code_to_id["de_DE"])
        decoded_out = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        # print(decoded_out)
        return decoded_out, batch['tgt_text']
    
    def test_epoch_end(self, results):
        total_score = 0
        count = 0
        # print(results)
        for batch_result in results:
            sys_outs = batch_result[0]
            refs = batch_result[1]
            rouge = Rouge()
            try:
                scores = rouge.get_scores(sys_outs, refs, avg=True)
            except:
                continue
            this_score = scores['rouge-1']['f']
            total_score += this_score
            count += 1
        if count == 0:
            print('Test Rouge-1:', 0)
        else:
            print('Test Rouge-1:', total_score/count)

    def predict_step(self, batch, batch_idx):
        self.model.model.set_test_mode()
        out = self.model(input_ids=batch['input_ids'], 
                        attention_mask=batch['attention_mask'])
        out = out * batch['attention_mask']
        # print(out)

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
                if this_score > self.chunk_cutoff:
                    out_str += ' |'
            outputs.append(out_str)
        return outputs

    def on_predict_epoch_end(self, results):
        with open(self.fname, "w") as of:
            outdict = {}
            count = 0
            for batch_result in results:
                for step_result in batch_result:
                    outdict[count] = step_result[0]
                    count += 1
            import json
            json.dump(outdict, of, indent=4)
    
    def set_fname(self, fname):
        self.fname = fname

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {'scheduler':get_linear_schedule_with_warmup(optimizer, self.total_steps*0.1, self.total_steps),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]


def main(hparams):
    tokenizer = AutoTokenizer.from_pretrained(hparams.model_name, src_lang="en_XX", tgt_lang="de_DE")
    litData = TranslationDataModule(tokenizer, hparams)

    if hparams.limit_train_batches <= 1:
        total_steps = math.ceil(len(litData.train_data) * hparams.limit_train_batches / hparams.batch_size) * hparams.epoch
    else:
        total_steps = hparams.limit_train_batches * hparams.epoch
    print("Total steps:", total_steps)

    hparams_hrnn = Hparams_hrnn(hparams.auxloss_coefficient, hparams.layers_ratio, hparams.heads_ratio, hparams.reweight_coefficient)

    if hparams.load_checkpoint:
        model = LitModel.load_from_checkpoint(checkpoint_path='lightning_logs/pretrain/epoch=02-train_loss=0.02-hrnn-translation.ckpt',
                                        learning_rate=hparams.learning_rate, model_name=hparams.model_name, total_steps=total_steps, hparams_hrnn=hparams_hrnn, tokenizer=tokenizer)
        # model = LitModel.load_from_checkpoint(checkpoint_path='lightning_logs/summarization/model-co_mse_0.10-co_attn_0.200-attmask_0.25-layers_0.50-nopretrain.ckpt',
        #                                 learning_rate=hparams.learning_rate, model_name=hparams.model_name, total_steps=total_steps, hparams_hrnn=hparams_hrnn)
    else:
        model = LitModel(learning_rate=hparams.learning_rate, model_name=hparams.model_name, total_steps=total_steps, hparams_hrnn=hparams_hrnn, tokenizer=tokenizer)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='./lightning_logs/translation',
        filename='model-co_mse_%.2f-co_attn_%.3f-attmask_%.2f-layers_%.2f-epoch_%.4f-bsz_%d-lr_%.6f-lv'%(hparams.auxloss_coefficient,
        hparams.reweight_coefficient, 
        hparams.heads_ratio,
        hparams.layers_ratio,
        hparams.limit_train_batches,
        hparams.batch_size,
        hparams.learning_rate,
        )
    )

    trainer = pl.Trainer(accelerator=hparams.accelerator, 
                        devices=hparams.devices, 
                        max_epochs=hparams.epoch,
                        limit_train_batches=hparams.limit_train_batches,
                        limit_predict_batches=hparams.limit_predict_batches,
                        # callbacks=[checkpoint_callback],
                        enable_checkpointing=False,
                        # callbacks=[LearningRateMonitor(logging_interval='step')],
                        # logger=False,
                        amp_backend='apex',
                        precision=16,
                        num_sanity_val_steps=0,
                        # accumulate_grad_batches=2,
                        )

    if not hparams.is_test:
        trainer.fit(model, litData)
    
    # trainer.test(model, litData)

    litData.predict_wmt = True
    model.set_fname(hparams.fname_indomain)
    trainer.predict(model, litData)

    litData.predict_wmt = False
    model.set_fname(hparams.fname_conll)
    trainer.predict(model, litData)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--devices", default=1)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--epoch", default=1, type=int)
    parser.add_argument("--model_name", default="facebook/mbart-large-50", type=str)
    parser.add_argument("--limit_train_batches", default=0.01, type=float)
    parser.add_argument("--limit_predict_batches", default=1.0, type=float)
    parser.add_argument("--is_test", action="store_true")
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--fname_indomain", default="./result_files/model_testset_result.json", type=str)
    parser.add_argument("--fname_conll", default="./result_files/model_conll2000_result.json", type=str)

    parser.add_argument("--auxloss_coefficient", default=0.1, type=float)
    parser.add_argument("--reweight_coefficient", default=0.2, type=float)
    parser.add_argument("--layers_ratio", default=0.5, type=float)
    parser.add_argument("--heads_ratio", default=0.5, type=float)
    parser.add_argument("--cut_rate", default=0.5, type=float)
    parser.add_argument("--predict_dataset", default="wmt", type=str)

    args = parser.parse_args()
    main(args)