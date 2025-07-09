import dataset.load_dataset
from torch.utils.data import DataLoader
raw_dataset_train, raw_dataset_test = dataset.load_dataset.create_preference_dataset('/home/yec/NetLLM/viewport_prediction/dataset', 'video_name')
dataloader_train = DataLoader(raw_dataset_train, batch_size=1, shuffle=False, pin_memory=True)

print(len(raw_dataset_train))
print(len(dataloader_train))

for step, (feature, label, video_info) in enumerate(dataloader_train): 
    print(step)
    print(feature)
    print(feature.shape)
    print(label)
    print(label.shape)
    print(video_info)
    
print(111)
