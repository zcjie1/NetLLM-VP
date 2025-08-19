import os
from torch.utils.data import Dataset
from config import cfg
import torch
import csv
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
from collections import Counter
import ast  # 新增: 用于解析字符串形式的列表
import numpy as np # 新增: 用于处理特征矩阵

class VideoPreferenceDataset(Dataset):
    def __init__(self, data_list):
        self.samples = data_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = torch.tensor(sample["features"], dtype=torch.float32)  # shape: [his_window, 152]
        label = torch.tensor(sample["label"], dtype=torch.long) # shape: [his_window, funture_dim]
        video_name = sample["video_name"]
        height = sample["height"]
        fps = sample["fps"]
        time = sample["time"]
        return features, label, (video_name, height, fps, time)


def split_dataset_per_video(sequence_samples, train_ratio=0.8, seed=42):
    random.seed(seed)
    
    # 按照视频名 + 时间点进行组织
    video_time_dict = defaultdict(lambda: defaultdict(list))
    for sample in sequence_samples:
        video = sample["video_name"]
        time = sample["time"]
        video_time_dict[video][time].append(sample)
    
    train_data = []
    val_data = []
    
    for video_name, time_dict in video_time_dict.items():
        all_times = list(time_dict.keys())
        assert len(all_times) == 10, f"视频 {video_name} 的时间点数量不是10个"
        
        selected_times = all_times[:]
        random.shuffle(selected_times)
        split_idx = int(train_ratio * len(selected_times))  # 8
        
        train_times = set(selected_times[:split_idx])
        val_times = set(selected_times[split_idx:])
        
        for t in train_times:
            train_data.extend(time_dict[t])
        for t in val_times:
            val_data.extend(time_dict[t])
    
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    return train_data, val_data


# split_type 划分数据集方式
# 按照video_name划分训练集/测试集
# 按照时间划分训练集/测试集
def create_preference_dataset(dataset_dir=cfg.dataset_dir, split_type=cfg.split_type, his_window=cfg.his_window, fut_window=cfg.fut_window, video_len=cfg.video_len):
    """
    读取 total_llm.csv, 构造包含历史 K s特征的训练/测试集。
    输入特征将是152维，包含30帧的5个特征+2个元数据。
    """
    csv_path = os.path.join(dataset_dir, cfg.dataset_name)
    data = []

    # 按行读取，存储为字典类型
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            sample = {
                "pixels": row["pixels"],
                "areas": row["areas"],
                "edges": row["edges"],
                "corner": row["corner"],
                "hist": row["hist"],
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
        "pixels", "areas", "edges", "corner", "hist"
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
            
            # --- 构建152维特征向量 ---
            # 初始化一个 30x5 的零矩阵，用于存放30帧的5个特征
            per_frame_features = np.zeros((30, 5), dtype=np.float32)

            # 解析5个特征的字符串列表
            feature_lists = []
            for k in feature_keys:
                try:
                    # 使用 ast.literal_eval 安全地将字符串转换为列表
                    feature_lists.append(ast.literal_eval(past_sample[k]))
                except (ValueError, SyntaxError):
                    # 如果解析失败（例如，空字符串），则视为空列表
                    feature_lists.append([])
            
            # print("feature_lists: ", feature_lists)
            # print("feature_lists shape: ", [len(l) for l in feature_lists])

            # 确定实际的帧数（取5个特征列表长度的最小值）
            num_frames = min(len(lst) for lst in feature_lists) if feature_lists else 0
            
            # 填充特征矩阵
            for i in range(num_frames):
                for j in range(len(feature_keys)):
                    per_frame_features[i, j] = feature_lists[j][i]
            
            # print(per_frame_features.shape)
            # print(per_frame_features)
            
            # 将30x5的矩阵展平为150维向量
            flat_features = per_frame_features.flatten().tolist()

            # print(flat_features)
            
            # 拼接 height 和 fps，形成最终的152维向量
            feature_vector = flat_features + [past_sample["height"], past_sample["fps"]]
            
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
    # 提取用于分层的标签（这里以未来标签序列第一个标签为准）
    stratify_labels = [s["label"][0] for s in sequence_samples]
    # 数据划分
    if split_type is None or split_type == "random_abs":
        train_samples, test_samples = train_test_split(
            sequence_samples,
            test_size=0.2,
            random_state=42,
            stratify=stratify_labels  # 按标签做分层，保证训练/测试集标签分布一致
        )
        train_data, test_data = train_samples, test_samples
    elif split_type == "abs_time":
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


if __name__ == "__main__":
    raw_dataset_train, raw_dataset_valid = create_preference_dataset(dataset_dir='./dataset',
                                                                        split_type=None,
                                                                        his_window=3,
                                                                        fut_window=3,
                                                                        video_len=10)