from dataset import CustomDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

transform = A.Compose([
    A.Resize(416, 416),
    A.Normalize(mean=[0,0,0], std=[1,1,1]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo'))

dataset = CustomDataset(transform=transform)

dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=8)

batch = next(iter(dataloader))

print(batch[0].shape)
print(batch[1].shape)
print(batch[2].shape)


