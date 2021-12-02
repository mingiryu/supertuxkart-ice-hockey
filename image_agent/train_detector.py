import torch
import numpy as np

from .detector import Detector, save_detector
from .utils import load_data
from . import dense_transforms

import torch.utils.tensorboard as tb
import torch.nn.functional as F


def train(args):
    print("starting setup...")
    from os import path
    model = Detector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """
    # hyperparameters
    epochs = int(args.epochs)
    batch_size = int(args.batch)
    num_workers = int(args.workers)
    lr = float(args.learn)
    train_path = 'data.pkl'

    # hooking up device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # creating loss
    alpha = 0.25
    gamma = 1
    loss = FocalLoss(alpha=alpha, gamma=gamma)

    # creating composition of image transforms
    transforms = dense_transforms.Compose([
        # dense_transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.1),
        # dense_transforms.RandomHorizontalFlip(),
        dense_transforms.ToTensor()
    ])

    # creating optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # sending model, loss, and data to device
    print('setup complete.')
    print('loading dataset...')
    # model = load_planner()
    model = model.to(device)
    loss = loss.to(device)
    train_data = load_data(train_path, batch_size=batch_size, num_workers=num_workers, transform=transforms, model=1)
    print('datset loaded.')


    # training
    print('beginning training...')
    global_step = 0
    lowest_loss = 10
    for epoch in range(epochs):
        model.train()
        losses = []
        for data, heatmap in train_data:
            data, heatmap = data.to(device), heatmap.to(device)


            # generate output, calculate loss gradient
            output = model(data)
            loss_value = loss(torch.squeeze(output), heatmap)
            losses.append(loss_value.detach().cpu().numpy())

            # logging
            # train_logger.add_scalar('loss', loss_value, global_step)
            # log(train_logger, data, heatmap, output, global_step)

            # step it up
            global_step += 1
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = np.mean(np.array(losses))
        print('Completed epoch %-3d\t avg detector loss: %-3f' % (epoch+1, avg_loss))
        if avg_loss < lowest_loss:
            save_detector(model)
            lowest_loss = avg_loss
    print('training complete.')

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # calculate the BCE loss first
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets)

        # now calculate the pt
        pt = torch.exp(-BCE_loss)

        # finally calculate the focal loss
        fl = self.alpha * ((1-pt)**self.gamma) * BCE_loss

        return torch.mean(fl)

def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-e', '--epochs', default=10)
    parser.add_argument('-b', '--batch', default=32)
    parser.add_argument('-w', '--workers', default=2)
    parser.add_argument('-lr', '--learn', default=1e-3)

    args = parser.parse_args()
    train(args)