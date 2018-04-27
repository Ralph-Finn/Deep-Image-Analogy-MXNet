from time import time
import matplotlib.pyplot as plt
import cv2
from PatchMatchCuda import PatchMatch
import numpy as np


def propagate(nnf, feat_A, feat_AP, feat_B, feat_BP, patch_size, iters=2, rand_search_radius=200):
    tic = time()
    pm = PatchMatch(nnf, feat_A, feat_AP, feat_B, feat_BP, patch_size)
    pm.propagate(iters=iters, rand_search_radius=rand_search_radius)
    print("propagate_time:{}".format(time() - tic))
    return pm


def init_nnf(size):
    nnf = np.zeros(shape=(2, size[0], size[1])).astype(np.int)
    nnf[0] = np.array([np.arange(size[0])] * size[1]).T
    nnf[1] = np.array([np.arange(size[1])] * size[0])
    nnf = nnf.transpose((1, 2, 0))
    return nnf


if __name__ == "__main__":
    # test here to see can pycuda work
    x = cv2.imread("data/ava.jpg")
    y = cv2.imread("data/mona.jpg")
    x = cv2.resize(x, (224, 224))
    y = cv2.resize(y, (224, 224))
    pm = PatchMatch(x, x, y, y, 3)
    pm.propagate(iters=10, rand_search_radius=32)
    plt.imshow(pm.visualize())
