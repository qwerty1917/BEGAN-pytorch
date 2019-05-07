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
    if not is_power_of_2(image_size) or image_size < 32:
        raise ValueError('image size should be 32, 64, 128, ...')

    if args.channel == 1:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            RandomNoise(mean=noise_mean, std=noise_std),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
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
        np_img = np.asarray(img)
        img_w = img.size[0]
        img_h = img.size[1]
        ch = len(img.getbands())

        # Add noise.
        noise = np.random.normal(self.mean, self.std, (img_h, img_w, ch))
        np_noisy = np.clip(np_img + noise, 0, 255)

        # Convert numpy array to PUL image.
        noisy = PIL.Image.fromarray(np.uint8(np_noisy))
        return noisy

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


if __name__ == '__main__':
    import argparse
    #os.chdir('..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--dset_dir', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    data_loader = return_data(args)
