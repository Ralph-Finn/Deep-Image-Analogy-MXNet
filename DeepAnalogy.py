from VGG19 import VGG19
from PatchMatchMxnet import init_nnf, propagate
import copy
from utils import *


def analogy(img_A, img_BP, config):
    # set basic param
    weights = config['weights']
    sizes = config['sizes']
    radius = config['radius']
    layers = config['layers']
    iters = config['iters']
    lr = config['lr']
    show_step = config['show_step']

    # compute 5 feature maps
    model = VGG19()
    data_A, data_A_size = model.get_features(img_tensor=img_A, layers=layers)
    data_AP = copy.deepcopy(data_A)
    data_BP, data_B_size = model.get_features(img_tensor=img_BP, layers=layers)
    data_B = copy.deepcopy(data_BP)
    for idx, layer_size in enumerate(data_A_size):
        print("layer_{}_size:".format(idx), layer_size)

    for curr_layer in range(5):
        if curr_layer == 0:
            ann_AB = init_nnf(data_A_size[curr_layer][2:])
            ann_BA = init_nnf(data_B_size[curr_layer][2:])
        else:
            ann_AB = pmAB.upsample_nnf(data_A_size[curr_layer][2])
            ann_BA = pmBA.upsample_nnf(data_B_size[curr_layer][2])

        # blend feature
        Ndata_A, response_A = normalize(data_A[curr_layer])
        Ndata_BP, response_BP = normalize(data_BP[curr_layer])

        data_AP[curr_layer] = blend(response_A, data_A[curr_layer], data_AP[curr_layer], weights[curr_layer])
        data_B[curr_layer] = blend(response_BP, data_BP[curr_layer], data_B[curr_layer], weights[curr_layer])

        Ndata_AP, _ = normalize(data_AP[curr_layer])
        Ndata_B, _ = normalize(data_B[curr_layer])

        # NNF search
        print("propagate_for_{}".format(curr_layer))

        pmAB = propagate(ann_AB, nd2np(Ndata_A), nd2np(Ndata_AP), nd2np(Ndata_B), nd2np(Ndata_BP),
                         sizes[curr_layer],
                         iters[curr_layer], radius[curr_layer])
        pmBA = propagate(ann_BA, nd2np(Ndata_BP), nd2np(Ndata_B), nd2np(Ndata_AP), nd2np(Ndata_A),
                         sizes[curr_layer],
                         iters[curr_layer], radius[curr_layer])

        if show_step:
            img_1 = pmAB.reconstruct_image(img_BP)
            img_2 = pmBA.reconstruct_image(img_A)
            output_img(post_process(img_1), post_process(img_2))

        if curr_layer < 4:
            # using backpropagation to approximate feature
            next_layer = curr_layer + 2
            data_AP_np = pmAB.reconstruct_image(nd2np(data_BP[next_layer]))
            data_B_np = pmBA.reconstruct_image(nd2np(data_A[next_layer]))
            target_BP_np = pmAB.reconstruct_image(nd2np(data_BP[curr_layer]))
            target_A_np = pmBA.reconstruct_image(nd2np(data_A[curr_layer]))

            print("deconvolution_for_{}".format(curr_layer))
            data_AP[curr_layer + 1] = model.get_deconvoluted_feat(np2nd(target_BP_np), curr_layer, np2nd(data_AP_np),
                                                                  lr=lr[curr_layer],
                                                                  iters=3000)
            data_B[curr_layer + 1] = model.get_deconvoluted_feat(np2nd(target_A_np), curr_layer, np2nd(data_B_np),
                                                                 lr=lr[curr_layer],
                                                                 iters=3000)

    print("reconstruction image")
    img_AP = pmAB.reconstruct_avg(img_BP, 5)   # size 5 is in paper
    img_B = pmBA.reconstruct_avg(img_A, 5)

    img_AP = post_process(img_AP)
    img_B = post_process(img_B)

    return img_AP, img_B
