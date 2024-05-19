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
from scipy import ndimage
import cv2
from detekce_incize_stehy import detekce


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
    plt.imshow(image[:,:,0], cmap='gray')
    plt.title('RGB - red')
    plt.axis('off')
    # plt.show()
    plt.subplot(3, 2, 3)
    plt.imshow(image[:,:,1], cmap='gray')
    plt.title('RGB - green')
    plt.axis('off')
    # plt.show()
    plt.subplot(3, 2, 5)
    plt.imshow(image[:,:,2], cmap='gray')
    plt.title('RGB - blue')
    plt.axis('off')
    # plt.show()

    hsv_img = rgb2hsv(image)*255
    plt.subplot(3, 2, 2)
    plt.imshow(hsv_img[:,:,0], cmap='gray')
    plt.title('HSV - hue')
    plt.axis('off')
    # plt.show()
    plt.subplot(3, 2, 4)
    plt.imshow(hsv_img[:,:,1], cmap='gray')
    plt.title('HSV - saturation')
    plt.axis('off')
    # plt.show()
    plt.subplot(3, 2, 6)
    plt.imshow(hsv_img[:,:,2], cmap='gray')
    plt.title('HSV - value')
    plt.axis('off')
    plt.show()

    gray = rgb2gray(image)
    plt.imshow(gray)
    plt.title('Šedotónový obrázek')
    plt.axis('off')
    plt.show()
    skimage.io.imsave("sedoton.png", img_as_ubyte(gray))
    return


def thresholding(image):
    gray = rgb2gray(image)
    blurred = ndimage.gaussian_filter(gray, sigma=1)
    # Aplikace adaptivního prahování s Gaussovým průměrem
    adaptive_thresh = cv2.adaptiveThreshold((blurred * 255).astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 91, 10)

    # První prahování
    _, binary_thresh1 = cv2.threshold(adaptive_thresh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Druhé prahování s jinou hodnotou prahu
    _, binary_thresh2 = cv2.threshold(adaptive_thresh, 200, 255, cv2.THRESH_BINARY)

    # Kombinace prahování
    combined_thresh = cv2.bitwise_or(binary_thresh1, binary_thresh2)

    labelled = morphology.label(combined_thresh)
    rp = measure.regionprops(labelled)
    size = [i.area for i in rp]
    size.sort()
    out = morphology.remove_small_objects(combined_thresh.astype(bool), min_size=size[-1] * 0.3)
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
    return np.argmin(vyrez)#0-8 mimo 4


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
    im = morphology.binary_dilation(im, kernel)

    labelled = morphology.label(im)
    rp = measure.regionprops(labelled)
    size = [i.area for i in rp]
    size.sort()
    if len(size)!=0:
        out = morphology.remove_small_objects(im.astype(bool), min_size=size[-1]*0.6)
        out = morphology.remove_small_objects(out.astype(bool), min_size=im.shape[0]*0.3*kernel.shape[0])
    else:
        out = im
    return out,incision


def vizualizace(img,kostra, incize):
    img[kostra,:] = [255,0,0]
    img[incize,:] = [0,255,0]
    return img


def process(images):
    output ={}
    for im in images:
        img_orig = skimage.io.imread("../images/incision_couples/" + im)
        if img_orig.shape[0] > img_orig.shape[1]:
           img_orig = skimage.transform.rotate(img_orig, 90, resize=True,)
        gray = rgb2gray(img_orig)
        blurred = ndimage.gaussian_filter(gray, sigma=1)
        # Aplikace adaptivního prahování s Gaussovým průměrem
        adaptive_thresh = cv2.adaptiveThreshold((blurred * 255).astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 91, 10)

        # Detekce čar pomocí Houghovy transformace
        lines = cv2.HoughLinesP(adaptive_thresh, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=15)

        if lines is None:
            output[im] = -1

        else:
            potential_stitch = detekce(img_orig)
            if potential_stitch == -1:
                output[im] = -1
            else:
                img = thresholding(img_orig)
                kost = kostra(img)
                stehy,incision = remove_incision(kost)
                labels, num = morphology.label(stehy, return_num=True)
                output[im] = num
                if args.v:
                    viz = vizualizace(img_orig, morphology.skeletonize(stehy), incision)
                    skimage.io.imsave("../images/visualization/" + im.split(".")[0]+".png", img_as_ubyte(viz))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=str)
    parser.add_argument('-v', action='store_true', required=False)
    parser.add_argument('input', nargs='+', type=str)
    args = parser.parse_args()
    out_file = args.out
    in_files = args.input

    output = process(in_files)#obrazky z args

    write_output("output.csv", output)


