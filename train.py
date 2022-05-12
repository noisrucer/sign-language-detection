import torch
import torch.optim as optim

from dataloader import CustomDataLoader
from yolov2 import Yolov2
from loss import Yolo_Loss
from metrics import mean_average_precision

device = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_EPOCHS = 100
LEARNING_RATE = 4e-6
BATCH_SIZE = 8
LOG_STEP = 1

def train():
    # [1] Dataloader
    train_loader = CustomDataLoader(
        data_dir='./data/train', label_dir='./data/labels', mode='train',
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0
    )

    val_loader = CustomDataLoader(
        data_dir='./data/val', label_dir='./data/labels', mode='val',
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0
    )

    # [2] Model
    model = Yolov2(n_classes=5)
    model = model.to(device)
    anchors = [(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053),
                (11.2364, 10.0071)]

    # [3] Loss Function
    loss_fn = Yolo_Loss()

    # [4] Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    total_loss = 0.0

    total_num_batches = len(train_loader)

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (images, gt_boxes, gt_labels) in enumerate(train_loader):
            images, gt_boxes, gt_labels = images.to(device), gt_boxes.to(device), gt_labels.to(device).type(torch.long)

            optimizer.zero_grad()
            preds = model(images)
            preds = torch.permute(preds, (0, 2, 3, 1))
            loss = loss_fn(preds, gt_boxes, gt_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % LOG_STEP == 0:
                print('Epoch: {} [{}/{}] Loss: {}'.format(
                    epoch+1,
                    batch_idx+1,
                    total_num_batches,
                    loss.item()
                ))

        total_loss /= total_num_batches

        # Calculate mAP
        train_mAP = mean_average_precision(train_loader, model, anchors)
        val_mAP = mean_average_precision(val_loader, model, anchors)
        print("-"*60)
        print("[Epoch {}] train mAP: {}".format(epoch+1, train_mAP))
        print("[Epoch {}] val mAP: {}".format(epoch+1, val_mAP))

if __name__=='__main__':
    train()
