import os
from pathlib import Path

import torch
import visdom
from torch import nn
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from datasets import return_data
from models.wgan.model import Discriminator
from utils import cuda

os.environ["CUDA_x_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


class WGAN(object):
    def __init__(self, args):
        # Misc
        self.args = args
        self.cuda = args.cuda and torch.cuda.is_available()
        self.sample_num = 100

        # Optimization
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.D_lr = args.D_lr
        self.G_lr = args.G_lr
        self.critic_iters = args.critic_iters
        self.global_epoch = 0
        self.global_iter = 0

        # Visualization
        self.env_name = args.env_name
        self.visdom = args.visdom
        self.port = args.port
        self.timestep = args.timestep
        self.output_dir = Path(args.output_dir).joinpath(args.env_name)
        self.visualization_init()

        # Network
        self.model_type = args.model_type
        self.fixed_z = Variable(cuda(self.sample_z(self.sample_num), self.cuda))
        self.ckpt_dir = Path(args.ckpt_dir).joinpath(args.env_name)
        self.load_ckpt = args.load_ckpt
        self.input_channel = args.channel
        self.multi_gpu = args.multi_gpu
        self.model_init()

        # Dataset
        self.dataset = args.dataset
        self.data_loader = return_data(args)

    def model_init(self):
        # TODO: WGAN model_init
        self.D = Discriminator(self.input_channel)
        self.G = Discriminator(self.input_channel)

        if self.cuda:
            self.D = cuda(self.D, self.cuda)
            self.G = cuda(self.G, self.cuda)

        if self.multi_gpu:
            self.D = nn.DataParallel(self.D).cuda()
            self.G = nn.DataParallel(self.G).cuda()

        if not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        if self.load_ckpt:
            self.load_checkpoint()

    def visualization_init(self):
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.visdom:
            self.viz_train_curves = visdom.Visdom(env=self.env_name + '/train_curves', port=self.port)
            self.viz_train_samples = visdom.Visdom(env=self.env_name + '/train_samples', port=self.port)
            self.viz_test_samples = visdom.Visdom(env=self.env_name + '/test_samples', port=self.port)
            self.viz_interpolations = visdom.Visdom(env=self.env_name + '/interpolations', port=self.port)
            self.win_moc = None

    def sample_z(self, batch_size=0, dist='uniform'):
        if batch_size == 0:
            batch_size = self.batch_size

        if dist == 'normal':
            return torch.randn(batch_size, 100)
        elif dist == 'uniform':
            return torch.rand(batch_size, 100).mul(2).add(-1)
        else:
            return None

    def sample_img(self, _type='fixed', nrow=10):
        self.set_mode('eval')

        if _type == 'fixed':
            z = self.fixed_z
        elif _type == 'random':
            z = self.sample_z(self.sample_num)
            z = Variable(cuda(z, self.cuda))
        else:
            self.set_mode('train')
            return

        samples = self.unscale(self.G(z))
        samples = samples.data.cpu()

        filename = self.output_dir.joinpath(_type+':'+str(self.global_iter)+'.jpg')
        grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)
        save_image(grid, filename=filename)
        if self.visdom:
            self.viz_test_samples.image(grid, opts=dict(title=str(filename), nrow=nrow, factor=2))

        self.set_mode('train')
        return grid

    def set_mode(self, mode='train'):
        if mode == 'train':
            self.G.train()
            self.D.train()
        elif mode == 'eval':
            self.G.eval()
            self.D.eval()
        else:
            raise ('mode error. It should be either train or eval')

    def unscale(self, tensor):
        return tensor.mul(0.5).add(0.5)

    def save_checkpoint(self, filename='ckpt.tar'):
        model_states = {'G': self.G.state_dict(),
                        'D': self.D.state_dict()}
        optim_states = {'G_optim': self.G_optim.state_dict(),
                        'D_optim': self.D_optim.state_dict()}
        states = {'iter': self.global_iter,
                  'epoch': self.global_epoch,
                  'args': self.args,
                  'win_moc': self.win_moc,
                  'fixed_z': self.fixed_z.data.cpu(),
                  'model_states': model_states,
                  'optim_states': optim_states}

        file_path = self.ckpt_dir.joinpath(filename)
        torch.save(states, file_path.open('wb+'))
        print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename='ckpt.tar'):
        file_path = self.ckpt_dir.joinpath(filename)
        if file_path.is_file():
            checkpoint = torch.load(file_path.open('rb'))
            self.global_iter = checkpoint['iter']
            self.global_epoch = checkpoint['epoch']
            self.win_moc = checkpoint['win_moc']
            self.fixed_z = checkpoint['fixed_z']
            self.fixed_z = Variable(cuda(self.fixed_z, self.cuda))
            self.G.load_state_dict(checkpoint['model_states']['G'])
            self.D.load_state_dict(checkpoint['model_states']['D'])
            self.G_optim.load_state_dict(checkpoint['optim_states']['G_optim'])
            self.D_optim.load_state_dict(checkpoint['optim_states']['D_optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

    def train(self):
        # WGAN train WIP
        pass
