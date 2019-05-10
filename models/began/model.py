"""model.py"""

import models.began.model_simple as simple
import models.began.model_skip_repeat as skip_repeat
import torch.nn as nn

import models.began.model_skip as skip


def encoder(_type, image_size, hidden_dim, n_filter, n_repeat, input_channel):
    if _type == 'simple':
        return simple.Encoder(image_size, hidden_dim, n_filter, n_repeat)
    elif _type == 'skip':
        return skip.Encoder(image_size, hidden_dim, n_filter, n_repeat)
    elif _type == 'skip_repeat':
        return skip_repeat.Encoder(image_size, hidden_dim, n_filter, n_repeat, input_channel)
    else:
        return None


def decoder(_type, image_size, hidden_dim, n_filter, n_repeat, input_channel):
    if _type == 'simple':
        return simple.Decoder(image_size, hidden_dim, n_filter, n_repeat)
    elif _type == 'skip':
        return skip.Decoder(image_size, hidden_dim, n_filter, n_repeat)
    elif _type == 'skip_repeat':
        return skip_repeat.Decoder(image_size, hidden_dim, n_filter, n_repeat, input_channel)
    else:
        return None


class Discriminator(nn.Module):
    def __init__(self, _type, image_size, hidden_dim, n_filter, n_repeat, input_channel):
        super(Discriminator, self).__init__()
        self.encode = encoder(_type, image_size, hidden_dim, n_filter, n_repeat, input_channel)
        self.decode = decoder(_type, image_size, hidden_dim, n_filter, n_repeat, input_channel)

    def weight_init(self, mean, std):
        self.encode.weight_init(mean, std)
        self.decode.weight_init(mean, std)

    def forward(self, image):
        out = self.encode(image)
        out = self.decode(out)

        return out


class Generator(nn.Module):
    def __init__(self, _type, image_size, hidden_dim, n_filter, n_repeat, input_channel):
        super(Generator, self).__init__()
        self.decode = decoder(_type, image_size, hidden_dim, n_filter, n_repeat, input_channel)

    def weight_init(self, mean, std):
        self.decode.weight_init(mean, std)

    def forward(self, h):
        out = self.decode(h)

        return out
