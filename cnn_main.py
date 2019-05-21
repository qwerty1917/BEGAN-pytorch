import argparse

import numpy as np
import torch

from solvers.solver_dcnn import DCNN

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

    if args.cnn_type == 'dcnn':
        net = DCNN(args)
    else:
        raise ValueError('cnn_type should be one of DCNN,')

    if args.mode == 'train':
        net.train()
    elif args.mode == 'eval':
        net.evaluate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DCNN')

    # Mode
    parser.add_argument('--mode', default='train', type=str, help='operation modes: train / eval')

    # Evaluation
    parser.add_argument('--eval_dir', default='cnn_eval', type=str, help='evaluation(test) result directory')

    # Optimization
    parser.add_argument('--epoch', default=20, type=int, help='epoch size')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate for the network')

    # Network
    parser.add_argument('--cnn_type', default='dcnn', type=str, help='CNN types : dcnn,')
    parser.add_argument('--load_ckpt', default=True, type=str2bool, help='load previous checkpoint')
    parser.add_argument('--ckpt_dir', default='cnn_checkpoint', type=str, help='weight directory')
    parser.add_argument('--image_size', default=32, type=int, help='image size')

    # Dataset
    parser.add_argument('--inter_fold_subject_shuffle', default=False, type=bool, help='subject shuffle inter-folds')
    parser.add_argument('--time_window', default=3, type=float, help='time window seconds')
    parser.add_argument('--augmentation', default=False, type=bool, help='GAN augmentation')
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='CIFAR10, CelebA')
    parser.add_argument('--num_workers', default=1, type=int, help='num_workers for data loader')
    parser.add_argument('--channel', default=3, type=int, help='input image channel')

    # Visualization
    parser.add_argument('--env_name', default='main', type=str, help='experiment name')
    parser.add_argument('--visdom', default=True, type=str2bool, help='enable visdom')
    parser.add_argument('--port', default=8081, type=int, help='visdom port number')
    parser.add_argument('--timestep', default=10, type=int, help='visdom curve time step')
    parser.add_argument('--output_dir', default='cnn_output', type=str, help='inter train result directory')

    # Misc
    parser.add_argument('--seed', default=1, type=int, help='random seed number')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--multi_gpu', default=False, type=str2bool, help='enable multi gpu')

    args = parser.parse_args()

    main(args)
