"""
The Patchmatch Algorithm. The actual algorithm is a nearly
line to line port of the original c++ version.
The distance calculation is different to leverage numpy's vectorized
operations.

This version uses 4 images instead of 2.
You can supply the same image twice to use patchmatch between 2 images.

"""
import os

package_directory = os.path.dirname(os.path.abspath(__file__))

import pycuda.driver as drv
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
from pycuda.compiler import SourceModule
import cv2


class PatchMatch(object):
    def __init__(self, nnf, a, aa, b, bb, patch_size):
        """
        Initialize Patchmatch Object.
        This method also randomizes the nnf , which will eventually
        be optimized.
        """
        assert a.shape == b.shape == aa.shape == bb.shape, "Dimensions were unequal for patch-matching input"
        self.A = a.copy(order='C')
        self.B = b.copy(order='C')
        self.AA = aa.copy(order='C')
        self.BB = bb.copy(order='C')
        self.patch_size = patch_size
        self.nnd = np.zeros(shape=(self.A.shape[0], self.A.shape[1])).astype(np.float32)  # the distance map for the nnf
        self.nnf = nnf.astype(np.int32)
        self.nnf = self.nnf.copy("C")

    def reconstruct_image(self, img_a):
        """
        Reconstruct image using the NNF and img_a.
        :param img_a: the patches to reconstruct from
        :return: reconstructed image
        """
        final_img = np.zeros_like(img_a, dtype=np.float)  # zeros_like get np.int if not set
        nnf = self.upsample_nnf(img_a.shape[0])
        for i in range(img_a.shape[0]):
            for j in range(img_a.shape[0]):
                x, y = nnf[i, j]
                final_img[i, j] = img_a[y, x]
        return final_img

    def upsample_nnf(self, size):
        """
        Upsample NNF based on size. It uses nearest neighbour interpolation
        :param size: INT size to upsample to.

        :return: upsampled NNF
        """
        temp = np.zeros((self.nnf.shape[0], self.nnf.shape[1], 3))

        for y in range(self.nnf.shape[0]):
            for x in range(self.nnf.shape[1]):
                temp[y][x] = [self.nnf[y][x][0], self.nnf[y][x][1], 0]

        img = np.zeros(shape=(size, size, 2), dtype=np.int)
        small_size = self.nnf.shape[0]
        aw_ratio = ((size) // small_size)
        ah_ratio = ((size) // small_size)

        temp = cv2.resize(temp, None, fx=aw_ratio, fy=aw_ratio, interpolation=cv2.INTER_NEAREST)

        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                pos = temp[i, j]
                img[i, j] = pos[0] * aw_ratio + (j - (j // aw_ratio) * aw_ratio), pos[1] * ah_ratio + (
                        i - (i // ah_ratio) * ah_ratio)

        return img

    def reconstruct_avg(self, img, patch_size=5):
        """
        Reconstruct image using average voting.
        :param img: the image to reconstruct from. Numpy array of dim H*W*3
        :param patch_size: the patch size to use

        :return: reconstructed image
        """
        size = (img.shape[0], img.shape[1])
        print("final_size_is:{}".format(img.shape))
        final = np.zeros(list(size) + [3, ])

        ah, aw = size
        bh, bw = size
        for i in range(size[0]):
            for j in range(size[1]):
                count = 0
                for di in range(-(patch_size // 2), (patch_size // 2 + 1)):
                    for dj in range(-(patch_size // 2), (patch_size // 2 + 1)):
                        if 0 <= (j + dj) < aw and 0 <= (i + di) < ah:
                            pos = self.nnf[i + di, j + dj]
                            if 0 <= (pos[0] - dj) < bw and 0 <= (pos[1] - di) < bh:
                                count += 1
                                final[i, j, :] += img[pos[1] - di, pos[0] - dj, :]
                if count > 0:
                    final[i, j] /= count

        return final

    def visualize(self):
        """
        Get the NNF visualisation
        :return: The RGB Matrix of the NNF
        """
        nnf = self.nnf

        img = np.zeros((nnf.shape[0], nnf.shape[1], 3), dtype=np.uint8)

        for i in range(nnf.shape[0]):
            for j in range(nnf.shape[1]):
                pos = nnf[i, j]
                img[i, j, 0] = int(255 * (pos[0] / self.B.shape[1]))
                img[i, j, 2] = int(255 * (pos[1] / self.B.shape[0]))

        return img

    def propagate(self, iters=2, rand_search_radius=500):
        """
        Optimize the NNF using PatchMatch Algorithm
        :param iters: number of iterations
        :param rand_search_radius: max radius to use in random search
        :return:
        """
        mod = SourceModule(open(os.path.join(package_directory, "patchmatch.cu")).read(), no_extern_c=True)
        patchmatch = mod.get_function("patch_match")

        rows = self.A.shape[0]
        cols = self.A.shape[1]
        channels = np.int32(self.A.shape[2])
        nnf_t = np.zeros(shape=(rows, cols), dtype=np.uint32)
        # threads = 20
        threads = 20

        def get_blocks_for_dim(dim, blocks):
            # if dim % blocks ==0:
            #    return dim//blocks
            return dim // blocks + 1

        patchmatch(
            drv.In(self.A),
            drv.In(self.AA),
            drv.In(self.B),
            drv.In(self.BB),
            drv.InOut(self.nnf),
            drv.InOut(nnf_t),
            drv.InOut(self.nnd),
            np.int32(rows),
            np.int32(cols),
            channels,
            np.int32(self.patch_size),
            np.int32(iters),
            np.int32(8),
            np.int32(rand_search_radius),
            block=(threads, threads, 1),
            grid=(get_blocks_for_dim(rows, threads),
                  get_blocks_for_dim(cols, threads)))
