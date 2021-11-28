# train():
#     model1 = planner()
#     model2 = planner()
#
#     # we got new training data, load models and continue training
#     model1 = load_model(model1)
#     model2 = load_model(model2)
#
#     while data in pkl:
#         extract image and label(enemy kart or ball info)
#             train attacker (model1) for ball info
#             train defender (model2) for enemy kart info

from .planner import Planner, save_model, load_model
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):
    from os import path
    model = Planner()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    # hyperparameters
    epochs = int(args.epochs)
    batch_size = int(args.batch)
    num_workers = int(args.workers)
    lr = float(args.learn)
    train_path = 'test_data'

    # hooking up device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # creating loss
    loss = torch.nn.MSELoss()

    # creating composition of image transforms
    transforms = dense_transforms.Compose([
        dense_transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.1),
        dense_transforms.RandomHorizontalFlip(),
        dense_transforms.ToTensor()
    ])

    # creating optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # sending model, loss, and data to device
    # model = load_model()
    model = model.to(device)
    loss = loss.to(device)
    train_data = load_data(train_path, batch_size=batch_size, num_workers=num_workers, transform=transforms)


    # training
    global_step = 0
    for epoch in range(epochs):
        model.train()
        for data, label in train_data:
            # print(label)
            data, label = data.to(device), label.to(device)

            # generate output, calculate loss gradient
            output = model(data)

            # loss is MSE of dist between kart pos and ball pos
            loss_value = loss(output, label)

            # step it up
            global_step += 1
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Completed epoch %-3d' % (epoch+1))
        save_model(model)

def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-e', '--epochs', default=10)
    parser.add_argument('-b', '--batch', default=128)
    parser.add_argument('-w', '--workers', default=2)
    parser.add_argument('-lr', '--learn', default=0.01)

    args = parser.parse_args()
    train(args)
