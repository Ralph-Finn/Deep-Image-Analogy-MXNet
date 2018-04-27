from mxnet import nd
import cv2
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt

ctx = mx.gpu(0)


def nd2np(x):
    x = x.reshape((x.shape[1], x.shape[2], x.shape[3]))
    x = x.transpose((1, 2, 0))
    return x.asnumpy()


def np2nd(x):
    x = x.transpose(2, 0, 1)
    x = nd.array(x).expand_dims(axis=0)
    return x.as_in_context(mx.gpu(0))


def blend(response, feature, recons, alpha=0.8, tau=0.05):
    """
    :param response:
    :param f_a: feature map (either F_A or F_BP)
    :param r_bp: reconstructed feature (R_BP or R_A)
    :param alpha: scalar balance the ratio of content and style in new feature map
    :param tau: threshold, default: 0.05 (suggested in paper)
    :return: (f_a*W + r_bp*(1-W)) where W=alpha*(response>tau)

    Following the official implementation, I replace the sigmoid function (stated in paper) with indicator function
    """
    weight = nd.array((response > tau)) * alpha
    # one can just use broadcasting to multiply
    # weight = weight.expand(1, f_a.size(1), weight.size(2), weight.size(3))
    weight = weight.as_in_context(ctx)
    f_new = feature * weight + recons * (1. - weight)
    return f_new


def normalize(feature_map):
    """

    :param feature_map: either F_a or F_bp
    :return:
    normalized feature map
    response
    """
    response = nd.sum(feature_map * feature_map, axis=1, keepdims=True)
    normed_feature_map = feature_map / nd.sqrt(response)
    # response should be scaled to (0, 1)
    response = (response - nd.min(response)) / (nd.max(response) - nd.min(response))
    # When the array is on a device, ordinary operations do not change the storage location of the array
    return normed_feature_map, response


def load_image(file):
    origin = cv2.imread(file)
    img = cv2.resize(origin, (224, 224), interpolation=cv2.INTER_CUBIC)
    return img


def post_process(img):
    img = img[:, :, ::-1] / 255.0
    return img.clip(0, 1)


def output_img(img_a, img_b):
    plt.subplot(1, 2, 1)
    plt.imshow(img_a)
    plt.subplot(1, 2, 2)
    plt.imshow(img_b)
    plt.show()
