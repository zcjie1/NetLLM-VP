import torch
import torch.nn as nn
from typing import *
from transformers.utils.dummy_pt_objects import PreTrainedModel
import os
from config import cfg
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, targets, weight=self.weight, reduction=self.reduction)
        return loss

class Pipeline(nn.Module):
    '''
    Sequence modeling pipeline for video preference prediction.
    '''
    def __init__(self,
                 plm: PreTrainedModel,
                 loss_func=None,
                 fut_window = None,
                 device='cuda',
                 embed_size=1024,
                 using_multimodal=False,
                 dataset=None):
        super().__init__()
        self.plm = plm
        self.using_multimodal = using_multimodal
        self.dataset = dataset
        self.device = device
        self.embed_size = embed_size
        # 预测未来多少个跨度的。当下是秒级别的
        self.fut_window_length = fut_window
        self.bce = False  # 是否使用BCELoss
        if loss_func == 'bce':
            print("Using BCELoss as the loss function.")
            loss_func = nn.BCELoss()
            self.bce = True
        elif loss_func == 'cross_entropy':
            print("Using CrossEntropyLoss as the loss function.")
            loss_func = nn.CrossEntropyLoss()
        elif loss_func is None:
            print("Using FocalLoss as the loss function.")
            loss_func = FocalLoss(gamma=2.0)
        
        self.input_embedding_net = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, self.embed_size),
            nn.LayerNorm(self.embed_size)  # 最终输出的embedding维度
        ).to(device)
        
        self.label_embed = nn.Embedding(2, 7).to(device)  # 用于标签的embedding
        self.loss_fct = loss_func
        self.fut_window = fut_window
        self.modules_except_plm = nn.ModuleList([  # used to save and load modules except plm
            self.input_embedding_net, self.label_embed, self.plm.networking_head
        ])

    def forward(self, batch, future, video_info, teacher_forcing=True, return_logits=False) -> torch.Tensor:
        """
        :param batch: history viewport trajectory7
        :param future: future viewport trajectory
        :param video_info: details information for current trajectory
        :param teacher_forcing: whether to use teacher forcing
        :param return_logits: whether to return logits for evaluation
        :return: loss value or (loss, logits)
        """
        if teacher_forcing:
            pred = self.teaching_forcing(batch, future, video_info)
        else:
            pred = self.auto_regressive(batch, future, video_info)
        
        gt = future.to(pred.device) # ground truth 真实标签

        if self.bce == True:
            pred_flat = pred.squeeze(-1) # [bs , seq_len]
            gt_flat = gt.float()
        else:
            pred_flat = pred.view(-1, 2)  # [bs * seq_len, 2]
            gt_flat = gt.view(-1).long()  # [bs * seq_len]
        loss = self.loss_fct(pred_flat, gt_flat) # loss_fct(input, target)

        if return_logits:
            return loss, pred_flat  # logits shape: [bs * seq_len, 2]
        else:
            return loss
    
    def auto_regressive(self, x, label, video_info) -> torch.Tensor:
        """
        auto-regressive generation
        
        :return: the loss value for training
        """
        # 这一步获取 his_wind, 保存历史几秒的信息
        seq_len = x.shape[1]
        batch_embeddings = []
        for i in range(seq_len):
            # 放入线性层，提取特征
            batch_embeddings.append(self.input_embedding_net(x[:, i, :]).unsqueeze(1))
        # seq_len维度拼接
        x = torch.cat(batch_embeddings, dim=1)
        outputlist = []
        for _ in range(self.fut_window_length):
            outputs = self.plm(inputs_embeds=x, attention_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.long, device=self.device))
            outputlist.append(outputs.logits)
            # outputs.logits.shape = [bs, 1, 1]

            probs = outputs.logits.squeeze(1).squeeze(-1)
            # 根据阈值0.5进行二分类，得到类别索引（LongTensor）
            class_idx = (probs > 0.5).long()  # 0 或 1
            # 传入embedding层，得到嵌入
            pred_emb = self.label_embed(class_idx)  # shape: [bs, embed_dim]
            pred_emb = self.input_embedding_net(pred_emb)  # 取最大值作为预测结果
            pred_emb = pred_emb.unsqueeze(1)
            x = torch.cat((x, pred_emb), dim=1)

        pred = torch.cat(outputlist, dim=1)
        return pred
    
    def teaching_forcing(self, x, future, video_info) -> torch.Tensor:
        """
        teaching-forcing generation

        :param x: history viewport trajectory
        :param future: future viewport trajectory
        :param video_info: details information for current trajectory
        :return: the return value by llm
        """
        # future 维度和x维度不一致，需要处理
        # future shape: [1, 3]
        future_emb = self.label_embed(future)  # shape: [1, 3, 7]
        x = torch.cat([x, future_emb], dim=1)  # shape: [1, 6, 7]
        seq_len = x.shape[1]
        batch_embeddings = []
        for i in range(seq_len):
            # batch_embeddings.append(self.embed_vp(self.conv1d(x[:, i, :]).view(1,256)).unsqueeze(1))
            batch_embeddings.append(self.input_embedding_net(x[:, i, :]).unsqueeze(1))
        x = torch.cat(batch_embeddings, dim=1)

        if self.using_multimodal:
            mapped_tensor = self.get_multimodal_information(video_info)
            x = torch.cat([mapped_tensor, x], dim=1)

        outputs = self.plm(inputs_embeds=x, attention_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.long, device=self.device), teacher_forcing=True)
        return outputs.logits
    
    def inference(self, feature, label, video_info) -> torch.Tensor:
        """
        Inference function. Use it for testing.
        """
        pred = self.auto_regressive(feature, label, video_info)
        gt = label.to(pred.device)
        return pred, gt
    
    # ToDo: 完成获取多模态特征这块
    # video_info 包含video_name、video_height、video_fps、video_time
    # def get_multimodal_information(self, video_info):
    #     """
    #     Get the corresponding image content features.
    #     Note that we use ViT to extract image features (the first output features of ViT that contains the overall information of the image).
    #     Since we use the frozen ViT for image feature extraction, we can actually use ViT to extract features first,
    #     then store all features into disk, and fetch them when needed.
    #     This way, we can avoid repeatedly using ViT to extract features for the same images.
    #     As a result, we can speed up the training process.
        
    #     TODO: Support on-the-fly image feature extraction with ViT.

    #     :param video_info: details information for current trajectory
    #     return: the embedding of the image
    #     """
    #     video_name = video_info[0][0]
    #     video_height = video_info[1].item()
    #     video_fps = video_info[2].item()
    #     video_time = video_info[3].item()
    #     # 定位到pth
    #     loaded_tensor_dict = torch.load(os.path.join(cfg.dataset_image_features, f'{video_name}/{video_height}/{video_fps}/feature_dict{(video_time)}.pth'))
    #     # 定位image index
    #     load_tensor = loaded_tensor_dict[f'{image_index}'].to(self.device)
    #     mapped_tensor = self.embed_multimodal(load_tensor)
    #     mapped_tensor = mapped_tensor.unsqueeze(1)
    #     return mapped_tensor