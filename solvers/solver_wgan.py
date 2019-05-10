import os
from pathlib import Path

import torch
import visdom
from torch.autograd import Variable

from datasets import return_data
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
        pass

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
        # TODO: WGAN sample_z
        pass

    # TODO: WGAN
