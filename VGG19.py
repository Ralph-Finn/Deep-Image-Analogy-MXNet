"""
This code is modified from harveyslash's work (https://github.com/harveyslash/Deep-Image-Analogy-PyTorch)
"""

import mxnet as mx
from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon import nn
from mxnet import nd
from mxnet import autograd
from time import time
from mxnet import optimizer
import sys


class VGG19:
    def __init__(self):
        vgg19_model = models.vgg19(pretrained=False)
        vgg19_model.load_params("model/vgg19.params", ctx=mx.cpu(0))  # pre-trained net is in cpu
        self.use_cuda = True
        self.mean = nd.array([0.485, 0.456, 0.406])
        self.std = nd.array([0.229, 0.224, 0.225])
        self.ctx = mx.gpu(0)
        self.model = self.get_model(vgg19_model)
        self.smooth = 0.5

    def get_model(self, pretrained_net):
        # We need to redefine a new network
        # because pre-trained structures cannot be read directly as "arrays."
        net = nn.Sequential()
        for i in range(40):
            net.add(pretrained_net.features[i])
        net.collect_params().reset_ctx(ctx=self.ctx)
        return net

    def preprocess(self, img):
        img = (nd.array(img).astype('float32') / 255.0 - self.mean) / self.std
        return img.transpose((2, 0, 1)).expand_dims(axis=0)

    def forward_subnet(self, x, start_layer, end_layer):
        for i, layer in enumerate(list(self.model)):
            if start_layer <= i <= end_layer:
                x = layer(x)
        return x

    def get_features(self, img_tensor, layers):
        img_tensor = self.preprocess(img_tensor)
        img_tensor = nd.array(img_tensor).copyto(self.ctx)
        features = []
        sizes = []
        x = img_tensor
        features.append(img_tensor)
        sizes.append(img_tensor.shape)
        for i in range(len(self.model)):
            x = self.model[i](x)
            if i in layers:
                features.append(x)
                sizes.append(x.shape)
        features.reverse()
        sizes.reverse()
        return features, sizes

    def get_deconvoluted_feat(self, feat, curr_layer, init=None, lr=10, iters=3000):
        # Deconvolution process: deconvolute the feature on one layer (e.g. L4) to the second last layer (e.g. L2)
        # and forward it to the last layer (e.g. L3).
        blob_layers = [29, 20, 11, 6, 1, -1]
        end_layer = blob_layers[curr_layer]
        mid_layer = blob_layers[curr_layer + 1]
        start_layer = blob_layers[curr_layer + 2] + 1
        # print("start:", start_layer, " mid:", mid_layer, " end", end_layer)
        # make sure the data is in GPU
        noise = init.copyto(self.ctx)
        target = feat.copyto(self.ctx)
        # get_sub_net
        net = nn.Sequential()
        for layer_num, layer in enumerate(list(self.model)):
            if start_layer <= layer_num <= end_layer:  # python simplified
                net.add(layer)
        net.collect_params().reset_ctx(ctx=self.ctx)

        def tv_loss(x):
            return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().sum() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().sum()

        def go(x):
            output = net(x)
            if curr_layer == 0:
                loss = (output - target).square().sum() + self.smooth * tv_loss(x)
            else:
                loss = (output - target).square().sum()
            return loss

        def train(x, lr, iters):
            tic = time()
            t = 1
            v = x.zeros_like()
            sqr = x.zeros_like()
            optim = optimizer.Adam(learning_rate=lr)
            for idx in range(iters):
                with autograd.record():
                    loss = go(x)
                loss.backward()
                optim.update(t, x, x.grad, [sqr, v])
                nd.waitall()  # TODO:it is a time cost operation
                t = t + 1
                sys.stdout.write('\r training..........%s%%' % (100 * idx // iters + 1))
                sys.stdout.flush()
            print("      all_train_time:", time() - tic)
            return x

        # begin training,just like style transfer
        noise.attach_grad()
        noise = train(noise, lr, iters)
        out = self.forward_subnet(noise, start_layer, mid_layer)
        return out
