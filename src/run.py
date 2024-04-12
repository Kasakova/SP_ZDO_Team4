import argparse
from skimage.color import rgb2hsv
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import try_all_threshold,threshold_otsu,threshold_yen,threshold_local
from skimage import measure, morphology,img_as_ubyte
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
    bin2 = image[:,:,2] > thresh
    im = np.logical_and(bin1,bin2)

    block_size = 11
    binary_adaptive = image[:,:,0] > threshold_local(image[:,:,0], block_size)
    im = np.logical_or(im,binary_adaptive)
    im = (im.astype(int)+1)%2

    labelled = morphology.label(im)
    rp = measure.regionprops(labelled)
    size = [i.area for i in rp]
    size.sort()
    try:
        out = morphology.remove_small_objects(im.astype(bool), min_size=size[-1]*0.8)
    except IndexError:
        out = im
    return out


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

    labelled = morphology.label(im)# TODO odstranit stehy, jejichz bounding boxy jsou vic na sirku nez na vysku
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
        img = thresholding(img_orig)
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


    write_output(out_file, output)

