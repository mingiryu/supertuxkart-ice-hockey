import numpy as np
import torch
import torch.utils.tensorboard as tb

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .dense_transforms import (
    Compose,
    RandomHorizontalFlip,
    ColorJitter,
    ToTensor,
)
from .planner import Planner, save_model


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15
DATASET_PATH = 'drive_data'


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, transform=ToTensor()):
        from PIL import Image
        from glob import glob
        from os import path

        self.data = []
        for f in glob(path.join(dataset_path, '*.csv')):
            i = Image.open(f.replace('.csv', '.png'))
            i.load()
            self.data.append(
                (i, np.loadtxt(f, dtype=np.float32, delimiter=','))
            )
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(*data)
        return data


def load_data(
    dataset_path=DATASET_PATH,
    transform=ToTensor(),
    num_workers=0,
    batch_size=128,
):
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )


def train(args):
    learning_rate = 1e-3
    weight_decay = 1e-5
    epochs = 80

    model = Planner()
    # model.load_state_dict(torch.load('homework/planner.th'))
    model.to(DEVICE)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    logger = tb.SummaryWriter('log/train', flush_secs=1)
    transform = Compose(
        [
            RandomHorizontalFlip(),
            ColorJitter(0.9, 0.9, 0.9, 0.1),
            ToTensor(),
        ]
    )
    data_loader = load_data('drive_data', transform=transform, num_workers=4)

    global_step = 0
    for epoch in range(epochs):
        with tqdm(data_loader, unit='batch') as tepoch:

            model.train()

            train_loss = []
            for data, target in tepoch:
                tepoch.set_description(f'epoch={epoch}')

                data, target = data.to(DEVICE), target.to(DEVICE)

                optimizer.zero_grad()
                output = model(data)

                loss = loss_fn(output, target)
                train_loss.append(loss.detach().cpu().numpy())

                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

            logger.add_scalar(
                'loss', np.mean(train_loss), global_step=global_step
            )

        global_step += 1
        save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
