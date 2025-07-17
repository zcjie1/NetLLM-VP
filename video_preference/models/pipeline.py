import torch
import torch.nn as nn
from typing import *
from transformers.utils.dummy_pt_objects import PreTrainedModel
import os
from config import cfg

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
        # 预测未来多少个跨度的。当下是：秒级别的
        self.fut_window_length = fut_window
        self.conv1d = nn.Sequential(nn.Conv1d(1, 256, 7), nn.LeakyReLU(), nn.Flatten()).to(device)
        self.embed_vp = nn.Linear(256, self.embed_size).to(device)
        self.output_mapper = nn.Sequential(
            nn.Linear(2, 256),   # 2分类输出映射到隐藏维度
            nn.ReLU(),
            nn.Linear(256, self.embed_size)  # 映射到和x同样的embedding维度
        ).to(device)
        self.label_embed = nn.Embedding(2, 7).to(device)  # 2分类标签嵌入

        self.embed_multimodal = nn.Linear(768, embed_size).to(device)  # 768 = ViT output feature size
        self.embed_ln = nn.LayerNorm(self.embed_size).to(device)

        self.loaded_tensor_cache = {}
        self.modules_except_plm = nn.ModuleList([  # used to save and load modules except plm
            self.embed_vp, self.embed_multimodal, self.embed_ln, self.conv1d, self.plm.networking_head
        ])

        if loss_func is None:
            loss_func = nn.CrossEntropyLoss()
        self.loss_fct = loss_func
        self.fut_window = fut_window

    def forward(self, batch, future, video_info, teacher_forcing=True) -> torch.Tensor:
        """
        :param batch: history viewport trajectory
        :param future: future viewport trajectory
        :param video_info: details information for current trajectory
        :return: the loss value for training
        """
        if teacher_forcing:
            pred = self.teaching_forcing(batch, future, video_info)
        else:
            pred = self.auto_regressive(batch, future, video_info)
        gt = future.to(pred.device)
        loss = self.loss_fct(pred, gt)
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
            # view(1,256) 需要改成 bs,256
            batch_embeddings.append(self.embed_vp(self.conv1d(x[:, i, :].unsqueeze(1)).view(1,256)).unsqueeze(1))
        x = torch.cat(batch_embeddings, dim=1)
        print(x.shape)
        if self.using_multimodal:  # we make using multimodal image features as an option, as not all datasets provide video information.
            mapped_tensor = self.get_multimodal_information(video_info)
            x = torch.cat([mapped_tensor, x], dim=1)

        x = self.embed_ln(x)

        outputlist = []
        for _ in range(self.fut_window_length):
            outputs = self.plm(inputs_embeds=x, attention_mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.long, device=self.device))
            print(outputs.logits.shape)
            outputlist.append(outputs.logits)
            print(x.shape)
            print(self.output_mapper(outputs.logits.squeeze(1)).unsqueeze(1).shape)
            # squeeze(1)去掉seq_len维度
            x = torch.cat((x, self.output_mapper(outputs.logits.squeeze(1)).unsqueeze(1)), dim=1)

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
        print(x.shape, future.shape)
        print(future)
        future_emb = self.label_embed(future)  # shape: [1, 3, 7]
        print(future_emb.shape)
        x = torch.cat([x, future_emb], dim=1)  # shape: [1, 6, 7]
        print(x.shape)
        # todo:7.15完成到这里，继续debug
        seq_len = x.shape[1]
        batch_embeddings = []
        for i in range(seq_len):
            batch_embeddings.append(self.embed_vp(self.conv1d(x[:, i, :]).view(1,256)).unsqueeze(1))
        x = torch.cat(batch_embeddings, dim=1)

        if self.using_multimodal:
            mapped_tensor = self.get_multimodal_information(video_info)
            x = torch.cat([mapped_tensor, x], dim=1)
        
        x = self.embed_ln(x)

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