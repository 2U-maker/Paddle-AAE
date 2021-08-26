import numpy as np
import math
import itertools
import paddle
import paddle.vision.transforms as T
import paddle.nn.functional as F
import paddle.nn as nn


def reparameterization(mu, logvar, latent_dim):
    std = paddle.exp(logvar / 2)
    sampled_z = paddle.to_tensor(np.random.normal(0, 1, (mu.shape[0], latent_dim)))
    z = sampled_z * std + mu
    return z


class Encoder(nn.Layer):
    def __init__(self, img_shape, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.BatchNorm1D(512),
            nn.LeakyReLU(0.2),
        )

        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def forward(self, img):
        img_flat = paddle.reshape(img, (img.shape[0], -1))
        x = self.model(img_flat)
        m = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(m, logvar, self.latent_dim)
        return z


class Decoder(nn.Layer):
    def __init__(self, img_shape, latent_dim):
        super(Decoder, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.BatchNorm1D(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.reshape((img_flat.shape[0], *self.img_shape))  # NCHW 
        return img


class Discriminator(nn.Layer):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity
