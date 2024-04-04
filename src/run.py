import argparse
import cv2
from skimage.color import rgb2hsv
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import try_all_threshold,threshold_otsu,threshold_yen,threshold_local
from skimage import measure, morphology,img_as_ubyte
from skimage.color import label2rgb
from scipy.ndimage import convolve
import os


def write_output(file, output):
    f = open(file, "w", encoding="UTF8")
    f.write("filename, n_stiches" + "\n")
    for item in output:
        f.write(item + ", " + str(output[item]) + "\n")
    f.close()
    return


def threshold_test(image):
    fig, ax = try_all_threshold(image[:,:,0], figsize=(10, 8), verbose=False)
    plt.show()
    fig, ax = try_all_threshold(image[:,:,1], figsize=(10, 8), verbose=False)
    plt.show()
    fig, ax = try_all_threshold(image[:,:,2], figsize=(10, 8), verbose=False)
    plt.show()

    hsv_img = rgb2hsv(image)*255

    fig, ax = try_all_threshold(hsv_img[:,:,0], figsize=(10, 8), verbose=False)
    plt.show()
    fig, ax = try_all_threshold(hsv_img[:,:,1], figsize=(10, 8), verbose=False)
    plt.show()
    fig, ax = try_all_threshold(hsv_img[:,:,2], figsize=(10, 8), verbose=False)
    plt.show()
    return


def thresholding(image):
    thresh = threshold_otsu(np.asarray(image[:,:,0]))
    bin1 = image[:,:,0] > thresh
    thresh = threshold_yen(image[:,:,1])
    bin2 = image[:,:,1] > thresh
    im = np.logical_and(bin1,bin2)
    # im = (im.astype(int)+1)%2

    block_size = 11
    binary_adaptive = image[:,:,0] > threshold_local(image[:,:,0], block_size)
    im = np.logical_or(im,binary_adaptive)
    im = (im.astype(int)+1)%2

    # plt.subplot(121)
    # plt.imshow(binary_adaptive)
    # plt.subplot(122)
    # plt.imshow(im)
    # plt.show()
    #TODO dilatace s necox1 kernelem na obe strany - spojit cary pokus?


    labelled = morphology.label(im)
    # plt.imshow(labelled)
    # plt.show()

    rp = measure.regionprops(labelled)
    size = [i.area for i in rp]
    size.sort()
    try:
        out = morphology.remove_small_objects(im.astype(bool), min_size=size[-2]+1)
    except IndexError:
        out = im
    # TODO rozhodnout podle velikosti kolik nechat objektu (kdyz se incision rozpuli)
    # plt.imshow(out)
    # plt.show()
    return out



def kostra(im):
    im = morphology.skeletonize(im)
    kernel = np.ones((5,5))
    im_neigh = convolve(im, kernel)
    im_neigh = morphology.skeletonize(im_neigh)
    # plt.imshow(im_neigh)
    # plt.show()
    return im_neigh

def remove_incision(im):
    yy, xx= np.nonzero(im)
    try:
        index_vpravo = np.argmax(xx) #pravy bod
        index_vlevo = np.argmin(xx) #levy bod
    except ValueError:
        return im #prazdny obraz, nebude se nic delat
    # print(xx[index_vlevo])
    # print(xx[index_vpravo])
    # print(yy[index_vlevo])
    # print(yy[index_vpravo])

    ix =min(xx[index_vlevo],xx[index_vpravo])
    ix2=max(xx[index_vlevo],xx[index_vpravo])
    iy =min(yy[index_vlevo],yy[index_vpravo])
    iy2=max(yy[index_vlevo],yy[index_vpravo])

    # plt.imshow(im)
    # plt.show()

    im[iy:iy2+1, ix:ix2+1] = 0 # TODO inteligentnejsi vymazani pouze incize

    # plt.imshow(im)
    # plt.show()

    kernel = np.ones((10,3))
    im = morphology.binary_dilation(im, kernel)
    # plt.imshow(im)
    # plt.show()

    labelled = morphology.label(im)
    rp = measure.regionprops(labelled)
    size = [i.area for i in rp]
    size.sort()
    if len(size)!=0:
        out = morphology.remove_small_objects(im.astype(bool), min_size=size[-1]*0.66)
    else:
        out = im

    # plt.imshow(out)
    # plt.show()

    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=str)
    parser.add_argument('-v', action='store_true', required=False)
    parser.add_argument('input', nargs='+', type=str)
    args = parser.parse_args()
    out_file = args.out
    in_files = args.input
    if args.v:
        print("visual mode")
    print(out_file)
    print(in_files)

    output = {}
    # for im in in_files:#obrazky z args
    #     img = skimage.io.imread("../images/incision_couples/" + im)
    #     # threshold_test(img)
    #     img = thresholding(img)
    #     img = kostra(img)
    #     img = remove_incision(img)
    #     labels, num = morphology.label(img, return_num = True)
    #     output[im] = num


    for im in os.listdir("../images/incision_couples/"):#vsechny obrazky
        img = skimage.io.imread("../images/incision_couples/" + im)
        img = thresholding(img)
        img = kostra(img)
        skimage.io.imsave("../images/skeletons/"+im, img_as_ubyte(img))
        img = remove_incision(img)
        skimage.io.imsave("../images/incisionless/"+im, img_as_ubyte(img))
        labels, num = morphology.label(img, return_num=True)
        output[im] = num


    # output = {"incision000.jpg": 5,"incision001.jpg": 2, "incision003.jpg": 0, "incision002.jpg": -1}
    write_output(out_file, output)
