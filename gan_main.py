"""main.py"""
import argparse

import numpy as np
import torch

from solvers.solver_began import BEGAN
from solvers.solver_wgan import WGAN
from utils import str2bool


def main(args):

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    if args.gan_type == 'began':
        net = BEGAN(args)
    elif args.gan_type == 'wgan':
        net = WGAN(args)
    else:
        raise ValueError('gan_type should be one of began, wgan')

    if args.mode == 'train':
        net.train()
    elif args.mode == 'eval':
        net.augment_img()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BEGAN')

    # Mode
    parser.add_argument('--mode', default='train', type=str, help='operation modes: train / eval')

    # Evaluation
    parser.add_argument('--augment_dir', default='augmentation', type=str, help='augmentation directory')
    parser.add_argument('--augment_num', default=10000, type=int, help='augmentation generate number')
    parser.add_argument('--best_ratio', default=0.01, type=float, help='best D loss sampling ratio')

    # Optimization
    parser.add_argument('--epoch', default=20, type=int, help='epoch size')
    parser.add_argument('--generator_iters', default=20000, type=int, help='number of iteration for generator in WGAN')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--D_lr', default=1e-4, type=float, help='learning rate for the Discriminator')
    parser.add_argument('--G_lr', default=1e-4, type=float, help='learning rate for the Generator')
    parser.add_argument('--lr_update_term', default=30, type=int, help='learning rate decaying(update) term')
    parser.add_argument('--gamma', default=0.5, type=float, help='equilibrium balance ratio')
    parser.add_argument('--lambda_k', default=0.001, type=float, help='the proportional gain of k')
    parser.add_argument('--critic_iters', default=5, type=int, help='update count of D while update G once')

    # Network
    parser.add_argument('--gan_type', default='began', type=str, help='GAN types : began, wgan')
    parser.add_argument('--model_type', default='skip_repeat', type=str, help='three types of models : simple, skip, skip_repeat')
    parser.add_argument('--n_filter', default=64, type=int, help='scaling unit of the number of filters')
    parser.add_argument('--n_repeat', default=2, type=int, help='repetition number of network layers')
    parser.add_argument('--hidden_dim', default=64, type=int, help='dimension of the hidden state')
    parser.add_argument('--load_ckpt', default=True, type=str2bool, help='load previous checkpoint')
    parser.add_argument('--ckpt_dir', default='checkpoint', type=str, help='weight directory')
    parser.add_argument('--image_size', default=32, type=int, help='image size')
    parser.add_argument('--weight_clipping_limit', default=0.01, type=float, help='WGAN-CP clipping limit')
    parser.add_argument('--model', default='default', type=str, help='WGAN model type. default or residual')

    # Dataset
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='CIFAR10, CelebA')
    parser.add_argument('--num_workers', default=1, type=int, help='num_workers for data loader')
    parser.add_argument('--channel', default=3, type=int, help='input image channel')
    parser.add_argument('--noise_mean', default=0, type=float, help='input image noise filter mean')
    parser.add_argument('--noise_std', default=0, type=float, help='input image noise filter std')
    parser.add_argument('--hide_range', default=0, type=float, help='Theater-hide range ratio')
    parser.add_argument('--checker_gap', default=5, type=int, help='Checker gap')
    parser.add_argument('--checker_intensity', default=0, type=int, help='Checker intensity')

    # Visualization
    parser.add_argument('--env_name', default='main', type=str, help='experiment name')
    parser.add_argument('--visdom', default=True, type=str2bool, help='enable visdom')
    parser.add_argument('--port', default=8081, type=int, help='visdom port number')
    parser.add_argument('--timestep', default=10, type=int, help='visdom curve time step')
    parser.add_argument('--output_dir', default='output', type=str, help='image output directory')

    # Misc
    parser.add_argument('--seed', default=1, type=int, help='random seed number')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--multi_gpu', default=False, type=str2bool, help='enable multi gpu')

    args = parser.parse_args()

    main(args)
