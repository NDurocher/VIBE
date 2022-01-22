import pickle
import pandas as pd
import numpy as np


def check_fixing_images():
    images = pickle.load(open('tmp_imag_q.obj', 'rb'))['images']
    # fileObj.close()
    # images = pd.read_pickle(r'tmp_images.obj')
    print(type(images))
    # print(images[3])
    images_pick = images[0]
    images_move = images[1]

    print(len(images))
    fixed_imgs = []
    for img in images:
        # print(img)
        if img:
            # print("img = ", img)
            for _img in img:
                if _img.any():
                    print("_img = ", _img)
                    fixed_imgs.append(_img)
    print(len(fixed_imgs))


def check_fixing_paths():
    qs = pickle.load(open('tmp_imag_q.obj', 'rb'))['qs']
    print(type(qs))
    print(qs[0])
    qs_pick = qs[0]
    qs_move = qs[1]

    print(len(qs))
    fixed_qs = []
    for q in qs:
        if q:
            print("q = ", q)
            for _q in q:
                print("_q = ", _q)
                fixed_qs.append(_q)

    print("fixed_qs", len(fixed_qs))
    print("fixed_qs", fixed_qs)
    # the goal is to add them together so the have similar shape

if __name__ == "__main__":

    check_fixing_images()
    # the goal is to add them together so the have similar shape

    # images = np.array(images, dtype=np.uint8)
    # print(type(images))
    # print(images.shape)
    # images = images.reshape(1,-1)
    # print(images.shape)
    # print(images)