from pathlib import Path

import torch
import visdom

from cnn_datasets import return_data


class DCNN(object):
    def __init__(self, args):

        # Mode
        self.mode = args.mode

        # Evaluation
        self.eval_dir = Path(args.eval_dir).joinpath(args.env_name)

        # Misc
        self.seed = args.seed
        self.cuda = args.cuda and torch.cuda.is_available()
        self.multi_gpu = args.multi_gpu

        # Optimization
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.lr = args.lr

        # Visualization
        self.env_name = args.env_name
        self.visdom = args.visdom
        self.port = args.port
        self.timestep = args.timestep
        self.output_dir = Path(args.output_dir).joinpath(args.env_name)
        self.visualization_init()

        # Network
        self.cnn_type = args.cnn_type
        self.load_ckpt = args.load_ckpt
        self.ckpt_dir = Path(args.ckpt_dir).joinpath(args.env_name)
        self.image_size = args.image_size
        self.model_init()

        # Dataset
        self.dataset = args.dataset
        self.data_loader = return_data(args)

    def model_init(self):
        # TODO: CNN model_init
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

    def save_checkpoint(self, file_name='ckpt.tar'):
        # TODO: CNN save_checkpoint
        pass

    def train(self):
        # TODO: CNN train
        pass


