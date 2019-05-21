"""datasets.py"""
from pathlib import Path

import PIL
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, LSUN, ImageFolder

__datasets__ = ['cifar10', 'celeba', 'lsun']


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    noise_mean = args.noise_mean
    noise_std = args.noise_std
    hide_range = args.hide_range
    checker_gap = args.checker_gap
    checker_intensity = args.checker_intensity
    if not is_power_of_2(image_size) or image_size < 32:
        raise ValueError('image size should be 32, 64, 128, ...')

    if args.channel == 1:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            RandomNoise(mean=noise_mean, std=noise_std),
            Theater(hide_range=hide_range),
            Checker(checker_gap=checker_gap, checker_intensity=checker_intensity),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            RandomNoise(mean=noise_mean, std=noise_std),
            Theater(hide_range=hide_range),
            Checker(checker_gap=checker_gap, checker_intensity=checker_intensity),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * args.channel, [0.5] * args.channel),
        ])

    if name.lower() == 'cifar10':
        root = Path(dset_dir).joinpath('CIFAR10')
        train_kwargs = {'root':root, 'train':True, 'transform':transform, 'download':True}
        dset = CIFAR10
    elif name.lower() == 'celeba':
        root = Path(dset_dir).joinpath('CelebA')
        train_kwargs = {'root':root, 'transform':transform}
        dset = ImageFolder
    elif name.lower() == 'lsun':
        raise NotImplementedError('{} is not supported yet'.format(name))
        root = Path(dset_dir).joinpath('LSUN')
        train_kwargs = {'root':str(root), 'classes':'train', 'transform':transform}
        dset = LSUN
    else:
        root = Path(dset_dir).joinpath(name)
        train_kwargs = {'root':root, 'transform':transform}
        dset = ImageFolder

    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = dict()
    data_loader['train'] = train_loader

    return data_loader


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


class Theater(object):
    """Add black belt to both 15% top and 15% bottom, 30% of total image
    
    Args:
        hide_range (float): ratio to hide with belt
    
    Return:
        PIL Image: noise added image.
    """
    def __init__(self, hide_range=0):
        self.hide_range = hide_range

    def __call__(self, img):
        """
        :param img: Image to add hiding filter 
        :return: Theater belted image.
        """

        # Convert PIL image to numpy.
        np_img = np.array(img)
        img_h = img.size[1]
        ch = len(img.getbands())

        # Add belt.
        top_end = int((img_h * self.hide_range)/2)
        bottom_start = img_h - int((img_h * self.hide_range)/2)

        if ch > 1:
            for ch_i in range(ch):
                np_img[:top_end, :, ch_i] = 0
                np_img[bottom_start:, :, ch_i] = 0
        else:
            np_img[:top_end, :] = 0
            np_img[bottom_start:, :] = 0

        # Convert numpy array to PIL image.
        belted_img = PIL.Image.fromarray(np_img.astype('uint8'))
        return belted_img

    def __repr__(self):
        return self.__class__.__name__ + '(hide_range={0})'.format(self.hide_range)


class Checker(object):
    """Add checker pattern on image to prevent loss cherry picking
    
    Args:
        checker_gap (int): pixel gap of checker pattern
        checker_intensity (int): checker intensity (of most dark park of checker)
    Return:
        PIL Image: noise added image.
    """
    def __init__(self, checker_gap=5, checker_intensity=0):
        self.checker_gap = checker_gap
        self.checker_intensity = checker_intensity

    def __call__(self, img):
        """
        :param img: Image to add checker pattern
        :return: Image added checker pattern
        """

        # Convert PIL image to numpy.
        np_img = np.array(img)
        img_w = img.size[0]
        img_h = img.size[1]
        ch = len(img.getbands())

        # Add checker
        half_intensity = self.checker_intensity//2
        if ch > 1:
            np_checker = np.zeros((img_h, img_w, ch))
            np_checker_row_1 = np.zeros((img_w, ch))
            np_checker_row_2 = np.zeros((img_w, ch))
        else:
            np_checker = np.zeros((img_h, img_w))
            np_checker_row_1 = np.zeros(img_w)
            np_checker_row_2 = np.zeros(img_w)

        for col_i in range(img_w):
            if col_i % (self.checker_gap * 2) == 0:
                if ch > 1:
                    np_checker_row_1[col_i:col_i + self.checker_gap, :] = half_intensity
                    np_checker_row_2[col_i:col_i + self.checker_gap, :] = half_intensity
                else:
                    np_checker_row_1[col_i:col_i + self.checker_gap] = half_intensity
                    np_checker_row_2[col_i:col_i + self.checker_gap] = half_intensity
        for row_i in range(img_h):
            if ch > 1:
                np_checker[row_i, :, :] += (np_checker_row_1 + np_checker_row_2)
            else:
                np_checker[row_i, :] += (np_checker_row_1 + np_checker_row_2)
            np_checker_row_1 = np.roll(np_checker_row_1, 1)
            np_checker_row_2 = np.roll(np_checker_row_2, -1)

        np_checkered = np.clip(np_img + np_checker, 0, 255)

        # Convert numpy array to PIL image.
        checkered_img = PIL.Image.fromarray(np_checkered.astype('uint8'))

        return checkered_img

    def __repr__(self):
        return self.__class__.__name__ + '(checker_gap={0}, checker_intensity={1})'.format(self.checker_gap, self.checker_intensity)




if __name__ == '__main__':
    import argparse
    #os.chdir('..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--dset_dir', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=64)

    # custom
    parser.add_argument('--channel', default=3, type=int, help='input image channel')
    parser.add_argument('--noise_mean', default=0, type=float, help='input image noise filter mean')
    parser.add_argument('--noise_std', default=10, type=float, help='input image noise filter std')
    parser.add_argument('--hide_range', default=0, type=float, help='Theater-hide range ratio')
    args = parser.parse_args()

    data_loader = return_data(args)
