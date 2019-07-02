from pathlib import Path

import PIL
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import random


def return_data(args):
    # TODO: cnn_datasets return_data
    train_dset_dir = args.train_dset_dir
    test_dset_dir = args.test_dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    time_window = args.time_window
    trivial_augmentation = args.trivial_augmentation
    sliding_augmentation = args.sliding_augmentation

    transform_list = [transforms.Resize((image_size, image_size))]

    if args.channel == 1:
        transform_list.append(transforms.Grayscale(num_output_channels=1))

    if sliding_augmentation:
        transform_list.append(RandomTimeWindow(time_window=time_window))
    else:
        transform_list.append(TimeWindow(time_window=time_window))

    if trivial_augmentation:
        trivial_transform_list = [
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(1, 1)),
            RandomNoise(mean=0, std=10),
        ]
        transform_list.append(transforms.RandomChoice(trivial_transform_list))

    transform_list.append(transforms.ToTensor())

    if args.channel == 1:
        transform_list.append(transforms.Normalize([0.5], [0.5]))
    else:
        transform_list.append(transforms.Normalize([0.5] * args.channel, [0.5] * args.channel))

    transform = transforms.Compose(transform_list)

    # if args.channel == 1:
    #     transform = transforms.Compose([
    #         transforms.Resize((image_size, image_size)),
    #         transforms.Grayscale(num_output_channels=1),
    #         TimeWindow(time_window=time_window),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5], [0.5]),
    #     ])
    # else:
    #     transform = transforms.Compose([
    #         transforms.Resize((image_size, image_size)),
    #         TimeWindow(time_window=time_window),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5] * args.channel, [0.5] * args.channel),
    #     ])

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


class RandomTimeWindow(object):
    def __init__(self, time_window=3):
        self.time_window = time_window
        self.max_time_window = 3

    def __call__(self, img):
        np_img = np.array(img)  # [row, col, ch]
        img_w = img.size[0]
        img_h = img.size[1]
        ch = len(img.getbands())

        trim_width = int((self.time_window/self.max_time_window) * img_w)

        crop_start = random.randint(0, img_w-trim_width)

        if ch > 1:
            np_trimmed = np_img[:, crop_start:crop_start+trim_width, :]
        else:
            np_trimmed = np_img[:, crop_start:crop_start+trim_width]

        trimmed_img = PIL.Image.fromarray(np_trimmed.astype('uint8'))

        return trimmed_img

    def __repr__(self):
        return self.__class__.__name__ + '(time_window={0})'.format(self.time_window)


class RandomNoise(object):
    """Add random noise on image.

    Args:
        mean (float): mean of noise
        std (float): std of noise

    Returns:
        PIL Image: noise added image.
    """

    def __init__(self, mean=0, std=0):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to add noise.

        Returns:
            PIL Image: Noise added image.
        """

        # Convert PIL image to numpy.
        np_img = np.array(img)
        img_w = img.size[0]
        img_h = img.size[1]
        ch = len(img.getbands())

        # Add noise.
        if ch > 1:
            noise = np.random.normal(self.mean, self.std, (img_h, img_w, ch))
        else:
            noise = np.random.normal(self.mean, self.std, (img_h, img_w))
        np_noisy = np.clip(np_img + noise, 0, 255)

        # Convert numpy array to PUL image.
        noisy = PIL.Image.fromarray(np_noisy.astype('uint8'))
        return noisy

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


