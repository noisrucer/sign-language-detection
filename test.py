from dataset import CustomDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.Resize(416, 416),
    A.Normalize(mean=[0,0,0], std=[1,1,1]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo'))
dataset = CustomDataset(transform=transform)

dataset[0]
