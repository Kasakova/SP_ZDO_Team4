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
from remove_incision import remove_incision
from scipy import ndimage
from skimage.io import imread
import cv2
def visualization():
    # Zobrazení původního a prahovaného obrázku
    plt.figure(figsize=(12, 8))  # Upravena velikost obrázku

    plt.subplot(2, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title('původní')
    plt.axis('off')

    plt.subplot(2, 4, 2)
    #plt.imshow(gray, cmap='gray')
    plt.title('šedotón')
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.imshow(img, cmap='gray')
    plt.title('filter')
    plt.axis('off')

    plt.subplot(2, 4, 6)
    plt.imshow(kostra_im, cmap='gray')
    plt.title('kostra')
    plt.axis('off')

    plt.subplot(2, 4, 7)
    plt.imshow(stehy , cmap='gray')
    plt.title('odstranění malých objektů')
    plt.axis('off')

    plt.subplot(2, 4, 8)
    plt.imshow(incision , cmap='gray')
    plt.title('odstranění malých objektů')
    plt.axis('off')

    plt.tight_layout()  # Upraví rozvržení tak, aby se obrázky nepřekrývaly
    plt.show()
    return None

def thresholding(image):
    grim = rgb2gray(image) # sedy vychazi nejlip lmao proc jsem se srala s jednotlivejma kanalama (aspoň odstavecek do docu)
    blurred = ndimage.gaussian_filter(grim, sigma=1)
    thresh = threshold_otsu(np.asarray(blurred))
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
def kostra(im):
    im = morphology.skeletonize(im)
    kernel = np.ones((5,5))
    im = morphology.binary_dilation(im, kernel)
    im = morphology.skeletonize(im)
    return im
def write_output(file, output):
    f = open(file, "w", encoding="UTF8")
    f.write("filename, n_stiches" + "\n")
    for item in output:
        f.write(item + ", " + str(output[item]) + "\n")
    f.close()
    return

images= os.listdir("../images/incision_couples/")
output ={}
for im in images:
    # Konverze do šedotónu
    image = skimage.io.imread("../images/incision_couples/" + im)

    img = thresholding(image)

    skimage.io.imsave("../images/threshold/"+im.split(".")[0]+".png", img_as_ubyte(img))

    kostra_im = kostra(img)
    skimage.io.imsave("../images/skeletons/"+im.split(".")[0]+".png", img_as_ubyte(kostra_im))
    stehy,incision = remove_incision(kostra_im)
    labels, num = morphology.label(stehy, return_num=True)
    skimage.io.imsave("../images/incisionless/"+im.split(".")[0]+".png", img_as_ubyte(stehy))
    output[im] = num
    
    write_output("output.csv" , output)
