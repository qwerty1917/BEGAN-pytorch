from pathlib import Path

import PIL
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def return_data(args):
    # TODO: cnn_datasets return_data
    train_dset_dir = args.train_dset_dir
    test_dset_dir = args.test_dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    time_window = args.time_window

    if args.channel == 1:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            TimeWindow(time_window=time_window),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            TimeWindow(time_window=time_window),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * args.channel, [0.5] * args.channel),
        ])

    train_root = Path(train_dset_dir)
    test_root = Path(test_dset_dir)
    train_kwargs = {'root': train_root, 'transform': transform}
    test_kwargs = {'root': test_root, 'transform': transform}
    dset = ImageFolder

    train_data = dset(**train_kwargs)
    test_data = dset(**test_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(test_data,
                             batch_size=252,
                             shuffle=True,
                             num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True)

    data_loader = dict()
    data_loader['train'] = train_loader
    data_loader['test'] = test_loader

    return data_loader


class TimeWindow(object):
    def __init__(self, time_window=3):
        self.time_window = time_window
        self.max_time_window = 3

    def __call__(self, img):
        np_img = np.array(img)  # [row, col, ch]
        img_w = img.size[0]
        img_h = img.size[1]
        ch = len(img.getbands())

        trim_width = int((self.time_window/self.max_time_window) * img_w)

        if ch > 1:
            np_trimmed = np_img[:, :trim_width, :]
        else:
            np_trimmed = np_img[:, :trim_width]

        trimmed_img = PIL.Image.fromarray(np_trimmed.astype('uint8'))

        return trimmed_img

    def __repr__(self):
        return self.__class__.__name__ + '(time_window={0})'.format(self.time_window)

