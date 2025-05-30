import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torchvision import transforms
from data.multi_mnist import MultiMNIST
from net.lenet import MultiLeNetR, MultiLeNetO
from pcgrad import PCGrad
from utils import create_logger
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import datetime
import os

# ------------------ CHANGE THE CONFIGURATION -------------
PATH = './dataset'
LR = 0.0005
BATCH_SIZE = 256
NUM_EPOCHS = 100
TASKS = ['R', 'L']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ---------------------------------------------------------


def main():
    accuracy = lambda logits, gt: ((logits.argmax(dim=-1) == gt).float()).mean()
    to_dev = lambda inp, dev: [x.to(dev) for x in inp]
    logger = create_logger('Main')

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"results_{now}"
    os.makedirs(result_dir, exist_ok=True)

    global_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dst = MultiMNIST(PATH, train=True, download=True, transform=global_transformer, multi=True)
    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    val_dst = MultiMNIST(PATH, train=False, download=True, transform=global_transformer, multi=True)
    val_loader = torch.utils.data.DataLoader(val_dst, batch_size=100, shuffle=True, num_workers=0)

    nets = {
        'rep': MultiLeNetR().to(DEVICE),
        'L': MultiLeNetO().to(DEVICE),
        'R': MultiLeNetO().to(DEVICE)
    }
    param = [p for v in nets.values() for p in list(v.parameters())]
    optimizer = torch.optim.Adam(param, lr=LR)
    optimizer = PCGrad(optimizer)

    train_losses_l, train_losses_r = [], []
    val_accs_l, val_accs_r = [], []

    for ep in range(NUM_EPOCHS):
        for net in nets.values():
            net.train()
        for batch in train_loader:
            mask = None
            optimizer.zero_grad()
            img, label_l, label_r = to_dev(batch, DEVICE)
            label_l = label_l.long()
            label_r = label_r.long()
            rep, mask = nets['rep'](img, mask)
            out_l, mask_l = nets['L'](rep, None)
            out_r, mask_r = nets['R'](rep, None)

            losses = [F.nll_loss(out_l, label_l), F.nll_loss(out_r, label_r)]
            optimizer.pc_backward(losses)
            optimizer.step()

        losses, acc = [], []
        for net in nets.values():
            net.eval()
        for batch in val_loader:
            img, label_l, label_r = to_dev(batch, DEVICE)
            label_l = label_l.long()
            label_r = label_r.long()
            mask = None
            rep, mask = nets['rep'](img, mask)
            out_l, mask_l = nets['L'](rep, None)
            out_r, mask_r = nets['R'](rep, None)

            losses.append([
                F.nll_loss(out_l, label_l).item(),
                F.nll_loss(out_r, label_r).item()
            ])
            acc.append([
                accuracy(out_l, label_l).item(),
                accuracy(out_r, label_r).item()])
        losses, acc = np.array(losses), np.array(acc)
        logger.info(f'epoches {ep}/{NUM_EPOCHS}: loss (L,R) = {losses[:,0].mean():.4f}, {losses[:,1].mean():.4f}')
        logger.info(f'epoches {ep}/{NUM_EPOCHS}: acc  (L,R) = {acc[:,0].mean():.3f}, {acc[:,1].mean():.3f}')

        train_losses_l.append(losses[:, 0].mean())
        train_losses_r.append(losses[:, 1].mean())
        val_accs_l.append(acc[:, 0].mean())
        val_accs_r.append(acc[:, 1].mean())

    # Plot individual loss
    plt.figure()
    plt.plot(train_losses_l)
    plt.title('Training Loss - Left Digit')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'{result_dir}/loss_left.png')

    plt.figure()
    plt.plot(train_losses_r)
    plt.title('Training Loss - Right Digit')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'{result_dir}/loss_right.png')

    # Plot individual accuracy
    plt.figure()
    plt.plot(val_accs_l)
    plt.title('Validation Accuracy - Left Digit')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(f'{result_dir}/acc_left.png')

    plt.figure()
    plt.plot(val_accs_r)
    plt.title('Validation Accuracy - Right Digit')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(f'{result_dir}/acc_right.png')

    # Show multiple batches of predictions
    for batch_idx in range(3):
        sample_batch = next(iter(val_loader))
        imgs, label_l, label_r = to_dev(sample_batch, DEVICE)
        label_l = label_l.long()
        label_r = label_r.long()
        rep, _ = nets['rep'](imgs, None)
        out_l, _ = nets['L'](rep, None)
        out_r, _ = nets['R'](rep, None)
        pred_l = out_l.argmax(dim=1)
        pred_r = out_r.argmax(dim=1)

        plt.figure(figsize=(12, 6))
        for i in range(8):
            plt.subplot(2, 4, i + 1)
            plt.imshow(imgs[i].cpu().squeeze(), cmap='gray')
            plt.title(f'True: {label_l[i].item()} {label_r[i].item()}\nPred: {pred_l[i].item()} {pred_r[i].item()}')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{result_dir}/sample_predictions_{batch_idx}.png')

    # Confusion matrices from last seen batch
    cm_l = confusion_matrix(label_l.cpu(), pred_l.cpu(), labels=list(range(10)))
    cm_r = confusion_matrix(label_r.cpu(), pred_r.cpu(), labels=list(range(10)))
    disp_l = ConfusionMatrixDisplay(confusion_matrix=cm_l, display_labels=list(range(10)))
    disp_r = ConfusionMatrixDisplay(confusion_matrix=cm_r, display_labels=list(range(10)))
    disp_l.plot()
    plt.title("Confusion Matrix - Left Digit")
    plt.savefig(f'{result_dir}/confusion_left.png')
    disp_r.plot()
    plt.title("Confusion Matrix - Right Digit")
    plt.savefig(f'{result_dir}/confusion_right.png')


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
