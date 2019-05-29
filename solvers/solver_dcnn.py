from pathlib import Path

import torch
import visdom
from torch import optim, nn
from torch.autograd import Variable

from cnn_datasets import return_data
from models.cnn.model_dcnn import Dcnn
from utils import cuda


## Weights init function, DCGAN use 0.02 std
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # Estimated variance, must be around 1
        m.weight.data.normal_(1.0, 0.02)
        # Estimated mean, must be around 0
        m.bias.data.fill_(0)


class DCNN(object):
    def __init__(self, args):
        self.args = args

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
        self.epoch_i = 0
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.global_iter = 0
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping_iter = args.early_stopping_iter

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
        self.input_channel = args.channel
        self.ckpt_dir = Path(args.ckpt_dir).joinpath(args.env_name)
        self.image_size = args.image_size
        self.model_init()

        # Dataset
        self.data_loader = return_data(args)

    def model_init(self):
        # TODO: CNN model_init
        self.C = Dcnn(self.input_channel)

        self.C.apply(weights_init)

        self.C_optim = optim.Adam(self.C.parameters(), lr=self.lr)

        if self.cuda:
            self.C = cuda(self.C, self.cuda)

        if self.multi_gpu:
            self.C = nn.DataParallel(self.C).cuda()

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

    def set_mode(self, mode='train'):
        if mode == 'train':
            self.C.train()
        elif mode == 'eval':
            self.C.eval()
        else:
            raise ('mode error. It should be either train or eval')

    def save_checkpoint(self, filename='ckpt_cnn.tar'):
        # TODO: CNN save_checkpoint
        model_states = {'C': self.C.state_dict()}
        optim_states = {'C_optim':self.C_optim.state_dict()}
        states = {'args': self.args,
                  'epoch': self.epoch,
                  'epoch_i': self.epoch_i,
                  'global_iter': self.global_iter,
                  'model_states': model_states,
                  'optim_states': optim_states}

        file_path = self.ckpt_dir.joinpath(filename)
        torch.save(states, file_path.open('wb+'))
        print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename='ckpt_cnn.tar'):
        file_path = self.ckpt_dir.joinpath(filename)
        if file_path.is_file():
            checkpoint = torch.load(file_path.open('rb'))
            self.args = checkpoint['args']
            self.epoch = checkpoint['epoch']
            self.epoch_i = checkpoint['epoch_i']
            self.global_iter = checkpoint['global_iter']
            self.C.load_state_dict(checkpoint['model_states']['C'])
            self.C_optim.load_state_dict(checkpoint['optim_states']['C_optim'])
            self.data_loader = return_data(self.args)
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

    def train(self):
        self.set_mode('train')
        min_loss = None
        min_loss_not_updated = 0
        early_stop = False

        while True:
            if self.epoch_i >= self.epoch or early_stop:
                break
            self.epoch_i += 1
            for i, (images, labels) in enumerate(self.data_loader['train']):
                images = Variable(cuda(images, self.cuda))
                labels = Variable(cuda(labels, self.cuda))

                self.global_iter += 1
                # Forward
                outputs = self.C(images)
                train_loss = self.criterion(outputs, labels)

                # Backward
                self.C_optim.zero_grad()
                train_loss.backward()
                self.C_optim.step()

                # train acc
                _, predicted = torch.max(outputs, 1)
                total = labels.size(0)
                correct = (predicted == labels).sum().item()
                train_acc = 100 * correct / total

                if self.global_iter % 1 == 0:
                    self.C.eval()
                    correct = 0
                    total = 0
                    for images, labels in self.data_loader['test']:
                        images = Variable(cuda(images, self.cuda))
                        labels = Variable(cuda(labels, self.cuda))
                        outputs = self.C(images)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        test_acc = 100 * correct / total
                        test_loss = self.criterion(outputs, labels)

                    print('Epoch [{}/{}], Iter [{}], train loss: {:.4f}, train acc.: {:.4f}, test loss:{:.4f}, test acc.: {:.4f} ({} / {}), min_loss_not_updated: {}'
                          .format(self.epoch_i + 1, self.epoch, i + 1, self.global_iter, train_loss.item(), train_acc, test_loss.item(), test_acc, correct, total, min_loss_not_updated))

                if self.global_iter % 10 == 0:
                    self.save_checkpoint()

                if min_loss is None:
                    min_loss = train_loss.item()
                elif train_loss.item() < min_loss:
                    min_loss = train_loss.item()
                    min_loss_not_updated = 0
                else:
                    min_loss_not_updated += 1

                if min_loss_not_updated >= self.early_stopping_iter:
                    early_stop = True

    def evaluate(self):
        self.load_checkpoint()
        self.set_mode('eval')

        self.C.eval()

        images, labels = next(iter(self.data_loader['test']))
        images = Variable(cuda(images, self.cuda))
        labels = Variable(cuda(labels, self.cuda))

        outputs = self.C(images)
        _, predicted = torch.max(outputs, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        eval_acc = 100 * correct / total
        test_loss = self.criterion(outputs, labels)
        env_name = self.args.env_name
        print("##### Env name: {} #####".format(env_name))
        print("Epoch: {}, iter: {}, test loss: {:.4f}, Test acc.: {:.4f}".format(self.epoch_i, self.global_iter, test_loss, eval_acc))






