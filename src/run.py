import argparse
from skimage.color import rgb2hsv
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import try_all_threshold,threshold_otsu,threshold_yen,threshold_local
from skimage import measure, morphology,img_as_ubyte
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage.color import rgb2gray
import os
from compare import comp


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

def channel_test(image):
    plt.subplot(3, 2, 1)
    plt.imshow(image[:,:,0])
    # plt.show()
    plt.subplot(3, 2, 2)
    plt.imshow(image[:,:,1])
    # plt.show()
    plt.subplot(3, 2, 3)
    plt.imshow(image[:,:,2])
    # plt.show()

    hsv_img = rgb2hsv(image)*255
    plt.subplot(3, 2, 4)
    plt.imshow(hsv_img[:,:,0])
    # plt.show()
    plt.subplot(3, 2, 5)
    plt.imshow(hsv_img[:,:,1])
    # plt.show()
    plt.subplot(3, 2, 6)
    plt.imshow(hsv_img[:,:,2])
    plt.show()
    return


def thresholding(image):
    # thresh = threshold_otsu(np.asarray(image[:,:,0]))
    # bin1 = image[:,:,0] > thresh
    # thresh = threshold_yen(image[:,:,1])
    # bin2 = image[:,:,1] > thresh
    # im = np.logical_and(bin1,bin2)
    #
    # hsv_img = rgb2hsv(image)*255
    # block_size = 13
    # binary_adaptive = hsv_img[:,:,2] > threshold_local(hsv_img[:,:,2], block_size)
    # im = np.logical_or(im,binary_adaptive)
    # im = (im.astype(int)+1)%2
    #
    # grim = rgb2gray(image) # sedy vychazi nejlip lmao proc jsem se srala s jednotlivejma kanalama (aspoň odstavecek do docu)
    #
    # thresh = threshold_otsu(np.asarray(grim))
    # bin1 = grim > thresh
    # thresh = threshold_yen(grim)
    # bin2 = grim > thresh
    # im = np.logical_and(bin1,bin2)
    #
    # block_size = 13
    # binary_adaptive = grim > threshold_local(grim, block_size)
    # im = np.logical_or(im,binary_adaptive)
    # im = (im.astype(int)+1)%2
    #
    # # kernel = np.ones((3,3))
    # # im = morphology.binary_dilation(im, kernel)
    #
    # labelled = morphology.label(im)
    # rp = measure.regionprops(labelled)
    # size = [i.area for i in rp]
    # size.sort()
    # try:
    #     out = morphology.remove_small_objects(im.astype(bool), min_size=size[-1]*0.8)
    # except IndexError:
    #     out = im
    #
    # kernel = np.ones((3,3))
    # out = morphology.binary_dilation(out, kernel)
    #
    # labelled = morphology.label(im)
    # rp = measure.regionprops(labelled)
    # size = [i.area for i in rp]
    # size.sort()
    # try:
    #     out = morphology.remove_small_objects(im.astype(bool), min_size=size[-1]*0.8)
    # except IndexError:
    #     out = im
    grim = rgb2gray(image) # sedy vychazi nejlip lmao proc jsem se srala s jednotlivejma kanalama (aspoň odstavecek do docu)

    # thresh = threshold_otsu(np.asarray(grim))
    # im = grim > thresh
    #
    # block_size = 13
    # binary_adaptive = grim > threshold_local(grim, block_size)
    # im = np.logical_or(im,binary_adaptive)
    # im = (im.astype(int)+1)%2

    thresh = threshold_otsu(np.asarray(grim))
    im = grim < thresh

    block_size = 15
    binary_adaptive = grim < threshold_local(grim, block_size)
    im = np.logical_and(im,binary_adaptive)
    # im = (im.astype(int)+1)%2

    labelled = morphology.label(im)
    rp = measure.regionprops(labelled)
    size = [i.area for i in rp]
    size.sort()
    try:
        im = morphology.remove_small_objects(im, min_size=size[-1] * 0.3)
    except IndexError:
        im = im

    kernel = np.ones((3,3))
    im = morphology.binary_dilation(im, kernel)

    labelled = morphology.label(im)
    rp = measure.regionprops(labelled)
    size = [i.area for i in rp]
    size.sort()
    try:
        out = morphology.remove_small_objects(im, min_size=size[-1]*0.8)
    except IndexError:
        out = im
    return out

def snake(img):
    s = np.linspace(0, 2 * np.pi, 400)
    r = img.shape[0]/2 + img.shape[0]/2 * np.sin(s)
    c = img.shape[1]/2 + img.shape[1]/2* np.cos(s)
    init = np.array([r, c]).T

    # snake
    snake = active_contour(gaussian(img, 3),
                           init, alpha=0.15, beta=30, gamma=0.00001,
                           coordinates='rc')
    # visualization
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])

    plt.show()
    return


def kostra(im):
    im = morphology.skeletonize(im)
    kernel = np.ones((5,5))
    im = morphology.binary_dilation(im, kernel)
    im = morphology.skeletonize(im)
    return im


def min_neighbor(im, x,y):
    vyrez = im[x-1:x+2,y-1:y+2].copy()
    vyrez[1,1] =100
    return np.argmin(vyrez)#0-8 mimo 4 #TODO preferovat cestu doprava?


def posun(index):
    i_posun=int(index%3) -1
    j_posun=int(index/3) -1
    return i_posun,j_posun


def find_path(im, bod1,bod2):
    im = im/255
    c = np.zeros_like(im)
    c[im == 0] = 100
    pozice = (bod1[0]+1,bod1[1]+1)
    bod2 = (bod2[0]+1,bod2[1]+1)
    c[pozice] = 1
    cesta = np.zeros((im.shape[0]+2,im.shape[1]+2))
    cesta = cesta +100
    cesta[1:-1,1:-1] = c
    while pozice != bod2:
        cesta[pozice] = cesta[pozice] + 1
        i,j = posun(min_neighbor(cesta, pozice[0],pozice[1]))
        pozice = (pozice[0]+j,pozice[1]+i)

    cesta = cesta[1:-1,1:-1]%2==1
    cesta = morphology.binary_dilation(cesta, np.ones([1,2]))
    cesta = morphology.remove_small_objects(cesta.astype(bool), min_size=3)

    return morphology.binary_erosion(cesta, np.ones([1,2]))


def remove_incision(im):
    labelled = morphology.label(im)
    rp = measure.regionprops(labelled)
    incision = np.zeros_like(im)
    for i in rp:
        y1,x1,y2,x2 = i.bbox#(min_row, min_col, max_row, max_col)
        yy, xx= np.nonzero(im[y1:y2,x1:x2])
        yy = yy+y1
        xx = xx+x1
        try:
            index_vpravo = np.argmax(xx) #pravy bod
            index_vlevo = np.argmin(xx) #levy bod
        except ValueError:
            return im #prazdny obraz, nebude se nic delat

        inc = find_path(img_as_ubyte(im), (yy[index_vlevo],xx[index_vlevo]), (yy[index_vpravo],xx[index_vpravo]))
        incision[inc] = 1

    im[incision] = 0

    kernel = np.ones((5,5))
    # kernel = morphology.diamond(1)

    im = morphology.binary_dilation(im, kernel)

    labelled = morphology.label(im)
    rp = measure.regionprops(labelled)

    size = [i.area for i in rp]
    size.sort()
    if len(size)!=0:
        out = morphology.remove_small_objects(im.astype(bool), min_size=size[-1]*0.5)
        out = morphology.remove_small_objects(out.astype(bool), min_size=im.shape[0]*0.3*kernel.shape[0])# TODO nejak odstranit naopak velky objekty?
    else:
        out = im
    return out,incision


def vizualizace(img,kostra, incize):
    img[kostra,:] = [255,0,0]
    img[incize,:] = [0,255,0]
    # plt.imshow(img)
    # plt.show()
    return img
def process(images):
    output ={}
    for im in images:
        img_orig = skimage.io.imread("../images/incision_couples/" + im)
        # img = snake(img_orig) # vraci nesmysly
        # threshold_test(img_orig)
        # channel_test(img_orig)
        img = thresholding(img_orig)
        skimage.io.imsave("../images/threshold/"+im.split(".")[0]+".png", img_as_ubyte(img))
        kost = kostra(img)
        skimage.io.imsave("../images/skeletons/"+im.split(".")[0]+".png", img_as_ubyte(kost))
        stehy,incision = remove_incision(kost)
        skimage.io.imsave("../images/incisionless/"+im.split(".")[0]+".png", img_as_ubyte(stehy))
        if args.v:
            viz = vizualizace(img_orig, morphology.skeletonize(stehy), incision)
            skimage.io.imsave("../images/visualization/" + im.split(".")[0]+".png", img_as_ubyte(viz))
        labels, num = morphology.label(stehy, return_num=True)
        output[im] = num
    return output

# TODO metoda hodne loops x moc velka cerna plocha -> -1,
# skimage.segmentation.clear_border

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=str)
    parser.add_argument('-v', action='store_true', required=False)
    parser.add_argument('input', nargs='+', type=str)
    args = parser.parse_args()
    out_file = args.out
    in_files = args.input

    # output = process(in_files)#obrazky z args
    output = process(os.listdir("../images/incision_couples/"))#vsechny obrazky


    anot = "anotace.csv"
    vyst = "output.csv"
    write_output(out_file , output)
    with open(anot, encoding="UTF8") as f1:
        real= f1.readlines()
    with open(vyst, encoding="UTF8") as f2:
        pred = f2.readlines()
    print(comp(real, pred))

