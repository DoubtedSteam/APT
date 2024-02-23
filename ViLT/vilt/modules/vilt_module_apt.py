import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer_apt as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils


class ViLTransformerSS(pl.LightningModule):
    def __init__(self, 
                 config, 
                 trainable=["classifier", "pooler", "token_type_embeddings", "rank_output"] + ['APT']
                ):
        super().__init__()
        self.save_hyperparameters()

        self.trainable = trainable

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        self.token4classifier = self.text_embeddings(torch.LongTensor([[101]]))
        self.token4classifier = nn.Parameter(self.token4classifier)

        # self.token4classifier = None

        # for n, p in self.named_parameters():
        #     if 'mm_tokens' in n:
        #         p = p * self.token4classifier.data[0]

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False

        self.trainable = trainable
        for n, p in self.named_parameters():
            # print(n)
            if not any(t in n for t in self.trainable):
                p.requires_grad = False
            else:
                print(n)

        orig_param_size = sum(p.numel() for p in self.parameters())
        trainable_size =  sum(p.numel() for p in self.parameters() if p.requires_grad)
        extra_param = sum(p.numel() for n, p in self.named_parameters() if "APT" in n)
        print('extra parameter:{}'.format(extra_param))
        print('trainable_size:{:.4f}%({}/{})'.format(trainable_size / orig_param_size * 100, trainable_size, orig_param_size))

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=True)

        self.text = torch.zeros(12)
        self.vis = torch.zeros(12)
        self.over = torch.zeros(12)
        self.count = 0

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)

        if self.token4classifier is not None:
            token4classifiers = self.token4classifier.repeat(text_embeds.shape[0], 1, 1)
            text_embeds = torch.cat([token4classifiers, text_embeds[:, 1:, :]], dim=1)

        if image_embeds is None and image_masks is None:
            img = batch[imgkey][0]
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds

        ex_co_maks = torch.bmm(co_masks.float().unsqueeze(2), co_masks.float().unsqueeze(1))
        ex_co_maks = torch.cat([ex_co_maks, torch.ones(ex_co_maks.shape[0], ex_co_maks.shape[1], 200).type(ex_co_maks.type())], dim=-1)

        # text_length = text_embeds.shape[1]
        # visual_length = image_embeds.shape[1]

        # self.count += co_embeds.shape[0]

        # attn_matrix = []
        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)
            # attn_matrix.append(_attn.mean(0))

        #     text_prompt_mask = torch.zeros_like(ex_co_maks)
        #     text_prompt_mask[:, :text_length, text_length+visual_length:] = 1
        #     text_prompt_mask = text_prompt_mask * ex_co_maks
        #     text_prompt_mask = text_prompt_mask.long()

        #     print(_attn.shape)
        #     print(text_prompt_mask.shape)

        #     text_prompt_weight = (_attn * text_prompt_mask).sum() / text_prompt_mask.sum() * 200 * x.shape[0]

        #     visual_prompt_mask = torch.zeros_like(ex_co_maks)
        #     visual_prompt_mask[:, text_length:text_length+visual_length, text_length+visual_length:] = 1
        #     visual_prompt_mask = visual_prompt_mask * ex_co_maks
        #     visual_prompt_mask = visual_prompt_mask.long()
        #     visual_prompt_weight = (_attn * visual_prompt_mask).sum() / visual_prompt_mask.sum() * 200 * x.shape[0]

        #     self.text[i] += text_prompt_weight.item()
        #     self.vis[i] += visual_prompt_weight.item()

        #     text_prompt_weight = (_attn * text_prompt_mask).sum(dim=1)[:, -200:] / text_prompt_mask.sum(dim=1)[:, -200:]
        #     visual_prompt_weight = (_attn * visual_prompt_mask).sum(dim=1)[:, -200:] / visual_prompt_mask.sum(dim=1)[:, -200:]
        #     self.over[i] += (text_prompt_weight * visual_prompt_weight).sum().item()

        # print(text_prompt_mask)
        # print(self.text / self.count)
        # print(self.vis / self.count)
        # print(self.over / self.count)

        # exit()

        # import cv2
        # import numpy as np
        # for i in range(len(self.transformer.blocks)):
        #     max_index_text = torch.sort(attn_matrix[i][0, :40], descending=True)[1]
        #     max_index_visual = torch.sort(attn_matrix[i][0, 40:-200], descending=True)[1] + text_embeds.shape[1]
        #     max_index_prompt = torch.sort(attn_matrix[i][0, -200:], descending=True)[1] + co_embeds.shape[1]

        #     range1 = 15
        #     range2 = 15

        #     max_index_text = max_index_text[:range1]
        #     max_index_visual = max_index_visual[:range2]
        #     max_index_prompt = max_index_prompt[:range1 + range2]
        #     max_index_text[-1] = 0

        #     max_index = torch.cat([max_index_text, max_index_visual, max_index_prompt], dim=-1)
        #     max_index = torch.sort(max_index)[0]

        #     attn = attn_matrix[i]
        #     # print(attn.shape)
        #     attn = attn[max_index[:range1 + range2]]
        #     attn = attn[:, max_index]
        #     # print(attn.shape)
        #     attn = attn - attn.min(dim=-1, keepdim=True)[0]
        #     attn = attn / attn.max(dim=-1, keepdim=True)[0]
        #     attn = (attn * 255)
        #     attn = attn.cpu().numpy().astype(np.uint8)
        #     attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)

        #     attn = cv2.resize(attn, (60 * 3, 30 * 3), interpolation=cv2.INTER_NEAREST)

        #     cv2.imwrite('visual/{}.png'.format(i), attn)

        # exit()

        x = self.transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        # vilt_utils.epoch_wrapup(self)
        pass

    # def validation_step(self, batch, batch_idx):
    #     vilt_utils.set_task(self)
    #     output = self(batch)

    # def validation_epoch_end(self, outs):
    #     # vilt_utils.epoch_wrapup(self)
    #     pass

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
            
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)