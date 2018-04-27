from DeepAnalogy import *
from time import time


def get_config(img_A, img_BP, weight):
    config = dict()
    config['img_A_path'] = img_A
    config['img_BP_path'] = img_BP
    config['layers'] = [29, 20, 11, 6, 1]
    config['iters'] = [10, 10, 10, 20, 10, 10]
    if weight == 2:
        config['weights'] = [1.0, 0.8, 0.7, 0.6, 0.1, 0.0]
    elif weight == 3:
        config['weights'] = [1.0, 0.9, 0.8, 0.7, 0.2, 0.0]
    config['sizes'] = [3, 3, 3, 5, 5, 3]
    config['radius'] = [32, 6, 6, 4, 4, 2]
    config['lr'] = [0.1, 0.1, 0.1, 0.1, 0.1]
    config['show_step'] = 1
    return config


if __name__ == "__main__":
    # settings
    config = get_config("data/ava.jpg", "data/mona.jpg", 2)
    # load images
    img_A = load_image(config['img_A_path'])
    img_BP = load_image(config['img_BP_path'])
    output_img(img_A[:, :, (2, 1, 0)], img_BP[:, :, (2, 1, 0)])
    # # Deep-Image-Analogy
    tic = time()
    img_AP, img_B = analogy(img_A, img_BP, config)
    print("all_analogy_time_is:{}".format(time() - tic))
    output_img(img_AP, img_B)
