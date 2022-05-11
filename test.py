from dataset import CustomDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataloader import CustomDataLoader
from yolov2 import Yolov2
from loss import Yolo_Loss
import torch

train_loader = CustomDataLoader(
    data_dir='./data/train', label_dir='./data/labels', mode='train',
    batch_size=8, shuffle=True, drop_last=False, num_workers=0
)

val_loader = CustomDataLoader(
    data_dir='./data/val', label_dir='./data/labels', mode='val',
    batch_size=8, shuffle=True, drop_last=False, num_workers=0
)

model = Yolov2(n_classes=5)
loss_fn = Yolo_Loss()

batch = next(iter(train_loader))
img, gt_boxes, gt_labels = batch

out = model(img)
out = torch.permute(out, (0, 2, 3, 1)) # (B, 50, 13, 13)
print("out shape: {}".format(out.shape))

loss = loss_fn(out, gt_boxes, gt_labels)
print("loss: {}".format(loss))



