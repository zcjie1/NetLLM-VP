import os
from torch.utils.data import Dataset
from config import cfg
import torch
import csv

class VideoPreferenceDataset(Dataset):
    def __init__(self, data_list):
        self.samples = data_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = torch.tensor(sample["features"], dtype=torch.float32)  # shape: [his_window, feature_dim]
        label = torch.tensor(sample["label"], dtype=torch.long)
        video_name = sample["video_name"]
        height = sample["height"]
        fps = sample["fps"]
        time = sample["time"]
        return features, label, (video_name, height, fps, time)

# split_type 划分数据集方式
# 按照video_name划分训练集/测试集
# 按照时间划分训练集/测试集
def create_preference_dataset(dataset_dir=cfg.dataset_dir, split_type=cfg.split_type, his_window=cfg.his_window, fut_window=cfg.fut_window, video_len=cfg.video_len):
    """
    读取 total_llm.csv, 构造包含历史 K s特征的训练/测试集。
    当 time < K 时，使用 time=0 的数据重复填充。
    """
    csv_path = os.path.join(dataset_dir, cfg.dataset_name)
    data = []

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            sample = {
                "pixel_diff": float(row["pixel diff"]),
                "area_diff": float(row["area diff"]),
                "edge_diff": float(row["edge diff"]),
                "hist_diff": float(row["hist diff"]),
                "surf_diff": float(row["surf diff"]),
                "height": float(row["height"]),
                "fps": float(row["fps"]),
                "label": int(row["label"]),
                "video_name": row["video name"],
                "time": int(float(row["time"])),  # 转为整数秒
            }
            data.append(sample)

    # 构建索引
    video_time_index = {}
    for sample in data:
        key = (sample["video_name"], sample["time"], sample["height"], sample["fps"])
        video_time_index[key] = sample

    # 提取特征字段
    feature_keys = [
        "pixel_diff", "area_diff", "edge_diff", "hist_diff",
        "surf_diff", "height", "fps"
    ]

    # 构建序列样本
    sequence_samples = []
    for sample in data:
        video = sample["video_name"]
        t = sample["time"]
        height = sample["height"]
        fps = sample["fps"]

        feature_sequence = []
        # 构造特征序列（过去his_win个）
        # t - (his_window - 1) 一直到 t
        for offset in reversed(range(his_window)):
            tt = (t - offset) % video_len
            key = (video, tt, height, fps)
            past_sample = video_time_index.get(key)
            feature_vector = [past_sample[k] for k in feature_keys]
            feature_sequence.append(feature_vector)
        
        # 构造未来标签序列，预测未来的（fut_wind）个label
        label_sequence = []
        for offset in range(fut_window):
            tt = (t + offset) % video_len
            key = (video, tt, height, fps)
            future_sample = video_time_index.get(key)
            label_sequence.append(future_sample["label"])
    
        sequence_samples.append({
            "features": feature_sequence,
            "label": label_sequence,
            "video_name": video,
            "time": t,
            "height": height,
            "fps": fps
        })

    # 数据划分
    if split_type is None or split_type == "time":
        train_data = [s for s in sequence_samples if s["time"] <= 7]
        test_data = [s for s in sequence_samples if s["time"] > 7]
    elif split_type == "video_name":
        all_video_names = sorted({s["video_name"] for s in sequence_samples})
        split_idx = int(0.8 * len(all_video_names))
        train_videos = set(all_video_names[:split_idx])
        train_data = [s for s in sequence_samples if s["video_name"] in train_videos]
        test_data = [s for s in sequence_samples if s["video_name"] not in train_videos]
    else:
        raise NotImplementedError("split_type must be None, 'time', or 'video_name'.")

    return VideoPreferenceDataset(train_data), VideoPreferenceDataset(test_data)