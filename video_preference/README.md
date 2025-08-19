# 视频偏好模块

## Pipeline设计

输入x维度[bs,seq_len,feature_dim]

### auto_regressive

1. 对每个时间序列的x进行cnn->linear，按照时间维度进行拼接
2. 拼接多模态信息
3. 按照未来窗口长度使用plm进行预测
   1. 得到预测结果
   2. 将预测结果拼接入x继续进行预测

### teaching_force

1. 直接拿到x和ground truth，将x和ground truth进行拼接
2. 和auto_regressive不同的是，一个使用自己预测出来的值拼接到x上，一个使用ground truth拼接到x上

##  video_perference命令

```sh
python3 run_plm.py --adapt --his-window 3 --fut-window 3 --plm-type llama --plm-size base --epochs 20 --bs 1 --lr 5e-5 --device cuda:0 --steps-per-valid 500 --save-checkpoint-per-epoch 1 --rank 8 --scheduled-sampling --video-len 10 --resume
```

```sh
python3 run_plm.py --test --his-window 3 --fut-window 3 --plm-type llama --plm-size base --epochs 20 --bs 1 --lr 5e-5 --device cuda:0 --steps-per-valid 500 --save-checkpoint-per-epoch 1 --rank 8 --scheduled-sampling --video-len 10
```

##  video_perference在线测试命令

```sh
python run_plm_zcj.py --online --his-window 3 --fut-window 3 --plm-type llama --plm-size base --epochs 100 --bs 1 --lr 0.0002 --grad-accum-steps 32 --device cuda:0 --steps-per-valid 5000 --save-checkpoint-per-epoch 1 --rank 32 --scheduled-sampling --video-len 10
```

## VIT多模态特征提取模块

1. tools/get_source_video.py：将数据集里的源视频(10s 1080p30fps)的视频拷贝到指定位置
2. tools/get_video_images.py：使用ffmpeg提取视频的每一帧的图像并且保存
3. dataset/extract_feateure.py：仿照viewpoint_predicition模块，将视频每1秒的30帧提取出来
4. dataset/video_feature：存放20个视频的VIT特征pth文件


1. 对齐小模型
2. 不做特征提取和特征量化
3. mixup论文的方法 

# 现在做的 75%
一. embedding -> plm 4096 -> NETWORKING HEAD(256 -> 二.)
networking head(2 linear) 256 hidden size 
7 -> hidden 1-> 4096-> hidden size 2 = 256 -> 2 

# 之前做的  87.5%
二. 2 liner-linear  32 hidden size
input -> hidden -> 1 (>0.5  1  <0.5 1)
7 -> 32 -> 1
