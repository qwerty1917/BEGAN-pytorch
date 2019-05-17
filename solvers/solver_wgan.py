import os
from collections import deque
from pathlib import Path

import torch
import visdom
import time
from torch import nn, optim
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from datasets import return_data
from models.wgan.model import Discriminator, Generator
from utils import cuda

os.environ["CUDA_x_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


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


class WGAN(object):
    def __init__(self, args):

        # Mode
        self.mode = args.mode

        # Evaluation
        self.augment_dir = Path(args.augment_dir).joinpath(args.env_name)
        self.augment_num = args.augment_num
        self.best_ratio = args.best_ratio

        # Misc
        self.args = args
        self.cuda = args.cuda and torch.cuda.is_available()
        self.sample_num = 100

        # Optimization
        self.epoch = args.epoch
        self.generator_iters = args.generator_iters
        self.batch_size = args.batch_size
        self.D_lr = args.D_lr
        self.G_lr = args.G_lr
        self.critic_iters = args.critic_iters
        self.generator_iter = 0
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
        self.weight_clipping_limit = args.weight_clipping_limit
        self.model_init()

        # Dataset
        self.dataset = args.dataset
        self.data_loader = return_data(args)

    def model_init(self):
        # TODO: WGAN model_init
        self.D = Discriminator(self.input_channel)
        self.G = Generator(self.input_channel)

        self.D.apply(weights_init)
        self.G.apply(weights_init)

        self.D_optim = optim.RMSprop(self.D.parameters(), lr=self.D_lr)
        self.G_optim = optim.RMSprop(self.G.parameters(), lr=self.G_lr)

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

    def sample_z(self, batch_size=0, dist='normal'):
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

        z = torch.unsqueeze(torch.unsqueeze(z, -1), -1)

        samples = self.unscale(self.G(z))
        samples = samples.data.cpu()

        filename = self.output_dir.joinpath(_type+':'+str(self.global_iter)+'.jpg')
        grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)
        save_image(grid, filename=filename)
        if self.visdom:
            self.viz_test_samples.image(grid, opts=dict(title=str(filename), nrow=nrow, factor=2))

        self.set_mode('train')
        return grid

    def augment_img(self):
        self.set_mode('eval')

        if not self.augment_dir.exists():
            self.augment_dir.mkdir(parents=True, exist_ok=True)

        generation_count = int(self.augment_num * (1 / self.best_ratio))

        memory_limit_count = 1000

        batch_count_list = [memory_limit_count]*(generation_count//memory_limit_count)
        batch_count_list.append(generation_count % memory_limit_count)

        total_augmented_count = 0
        for batch_i, batch_generation_count in enumerate(batch_count_list):
            if batch_generation_count <= 0:
                continue

            if batch_i < len(batch_count_list) - 1:
                batch_best_count = int(batch_generation_count * self.best_ratio)
            else:
                batch_best_count = self.augment_num - total_augmented_count

            z = self.sample_z(batch_generation_count)
            z = Variable(cuda(z, self.cuda))

            z = torch.unsqueeze(torch.unsqueeze(z, -1), -1)

            samples_pool = self.G(z)
            d_losses = self.D(samples_pool)

            best_indices = torch.topk(torch.squeeze(d_losses), batch_best_count, largest=False, sorted=False)[1]
            best_samples = samples_pool[best_indices]

            best_samples = self.unscale(best_samples)
            best_samples = best_samples.data.cpu()

            for sample_i, sample in enumerate(best_samples):
                total_augmented_count += 1
                filename = self.augment_dir.joinpath('augmented:' + str(total_augmented_count) + '.png')
                save_image(sample, filename=filename)

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
                  'g_iter': self.generator_iter,
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
            self.generator_iter = checkpoint['g_iter']
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

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images

    def train(self):
        self.set_mode('train')

        self.data = self.get_infinite_batches(self.data_loader['train'])

        one = torch.FloatTensor([1])
        mone = one * -1

        if self.cuda:
            one = one.cuda()
            mone = mone.cuda()

        for g_iter in range(self.generator_iters):

            ## Discriminator training
            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True

            self.generator_iter += 1
            e_elapsed = time.time()

            for d_iter in range(self.critic_iters):
                self.global_iter += 1
                self.D.zero_grad()

                # clamp parameters to a range [-c, c], c = weight_clipping_limit
                for p in self.D.parameters():
                    p.data.clamp_(-self.weight_clipping_limit, self.weight_clipping_limit)

                images = self.data.__next__()
                # Check for batch to have full batch_size
                if images.size()[0] != self.batch_size:
                    continue

                if self.cuda:
                    images = Variable(images.cuda())
                else:
                    images = Variable(images)

                # Discriminator Training with real image
                x_real = Variable(cuda(images, self.cuda))
                D_loss_real = self.D(x_real)
                D_loss_real = D_loss_real.mean(0).view(1)
                D_loss_real.backward(one)

                # TODO: github: line 166 / local began: line 230
                # Discriminator Training
                z = Variable(torch.randn(self.batch_size, 100, 1, 1))
                if self.cuda:
                    z = z.cuda()
                x_fake = self.G(z)
                D_loss_fake = self.D(x_fake)
                D_loss_fake = D_loss_fake.mean(0).view(1)
                D_loss_fake.backward(mone)

                D_loss = D_loss_fake - D_loss_real
                Wasserstein_D = D_loss_real - D_loss_fake
                self.D_optim.step()

            ## Generator training
            for p in self.D.parameters():
                p.requires_grad = False

            self.G.zero_grad()

            # Generator update
            # Compute loss with fake images
            z = Variable(torch.randn(self.batch_size, 100, 1, 1))
            if self.cuda:
                z = z.cuda()

            x_fake = self.G(z)
            G_loss = self.D(x_fake)
            G_loss = G_loss.mean().mean(0).view(1)
            G_loss.backward(one)
            G_cost = -G_loss
            self.G_optim.step()

            # Visualize process
            if self.visdom and ((self.global_iter <= 1000 and self.global_iter % 10 == 0) or self.global_iter % 200 == 0):
                self.viz_train_samples.images(
                    self.unscale(x_fake).data.cpu(),
                    opts=dict(title='x_fake'))
                self.viz_train_samples.images(
                    self.unscale(x_real).data.cpu(),
                    opts=dict(title='x_real'))

            if self.visdom and ((self.global_iter <= 1000 and self.global_iter % 10 == 0) or self.global_iter % 200 == 0):
                self.interpolation(self.fixed_z[0:1], self.fixed_z[1:2])
                self.sample_img('fixed')
                self.sample_img('random')
                self.save_checkpoint()

            # console output
            if self.global_iter % 10 == 0:
                print('generator_iter:{:d}, global_iter:{:d}'.format(self.generator_iter, self.global_iter))
                print('Wasserstein distance: {:.3f}, Loss D: {:.3f}, Loss G: {:.3f}, Loss D Real: {:.3f}, Loss D fake: {:.3f}'.
                      format(Wasserstein_D.data[0], D_loss.data[0], G_cost.data[0], D_loss_real.data[0], D_loss_fake.data[0]))

            e_elapsed = (time.time() - e_elapsed)
            print('generator_iter {:d}, [{:.2f}s]'.format(self.generator_iter, e_elapsed))

    def interpolation(self, z1, z2, n_step=10):
        self.set_mode('eval')
        filename = self.output_dir.joinpath('interpolation' + ':' + str(self.global_iter) + '.jpg')

        step_size = (z2 - z1) / (n_step + 1)
        buff = z1
        for i in range(1, n_step + 1):
            _next = z1 + step_size * (i)
            buff = torch.cat([buff, _next], dim=0)
        buff = torch.cat([buff, z2], dim=0)

        buff = torch.unsqueeze(torch.unsqueeze(buff, -1), -1)

        samples = self.unscale(self.G(buff))

        grid = make_grid(samples.data.cpu(), nrow=n_step + 2, padding=1, pad_value=0, normalize=False)
        save_image(grid, filename=filename)
        if self.visdom:
            self.viz_interpolations.image(grid, opts=dict(title=str(filename), factor=2))

        self.set_mode('train')






