from .planner import Planner, save_planner, load_planner
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):
    print("starting setup...")
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
    train_path = 'data.pkl'

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
    print('setup complete.')
    print('loading dataset...')
    # model = load_planner()
    model = model.to(device)
    loss = loss.to(device)
    train_data = load_data(train_path, batch_size=batch_size, num_workers=num_workers, transform=transforms, model=0)
    print('datset loaded.')


    # training
    print('beginning training...')
    global_step = 0
    lowest_loss = 10
    for epoch in range(epochs):
        model.train()
        losses = []
        for data, label in train_data:
            data, label = data.to(device), label.to(device)

            # generate output, calculate loss gradient
            output = model(data)

            # loss is MSE of dist between kart pos and ball pos
            loss_value = loss(output, label)
            losses.append(loss_value.detach().cpu().numpy())

            # step it up
            global_step += 1
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = np.mean(np.array(losses))
        print('Completed epoch %-3d\t avg planner loss: %-3f' % (epoch+1, avg_loss))
        if avg_loss < lowest_loss:
            save_planner(model)
            lowest_loss = avg_loss
    print('training complete.')

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
    parser.add_argument('-b', '--batch', default=32)
    parser.add_argument('-w', '--workers', default=2)
    parser.add_argument('-lr', '--learn', default=0.01)

    args = parser.parse_args()
    train(args)