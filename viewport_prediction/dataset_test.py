from dataset.load_dataset import create_dataset
from torch.utils.data import DataLoader


raw_dataset_train, raw_dataset_valid = create_dataset("Jin2022", his_window=5, 
                                                        fut_window=10, trim_head=15, trim_tail=30,
                                                        include=['train', 'valid'], frequency=5, step=15)

dataloader_train = DataLoader(raw_dataset_train, batch_size=1, shuffle=True, pin_memory=True)
dataloader_valid = DataLoader(raw_dataset_valid, batch_size=1, shuffle=False, pin_memory=True)
print(len(dataloader_train))
for step, (history, future, video_user_info) in enumerate(dataloader_train): 
    print(step)
    print(history)
    seq_len = history.shape[1]
    print(seq_len)
    print(history.shape)
    print(history[:, 0, :])
    print(history[:, 0, :].shape)
    print(future.shape)
    print(video_user_info)

print(111)


