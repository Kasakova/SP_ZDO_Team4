import cv2
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
from skimage import measure, morphology,img_as_ubyte
from scipy import ndimage
from skimage.morphology import diamond
from remove_incision import remove_incision
import os
import skimage.io
def kostra(im):
    im = morphology.skeletonize(im)
    kernel = np.ones((5,5))
    im = morphology.binary_dilation(im, kernel)
    im = morphology.skeletonize(im)
    return im


def visualize(image,gray,blurred,adaptive_thresh,combined_thresh,comb_ero,out,kostra_im,incision,stehy):
    # Zobrazení původního a prahovaného obrázku
    plt.figure(figsize=(12, 8))  # Upravena velikost obrázku

    plt.subplot(3, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title('původní')
    plt.axis('off')

    plt.subplot(3, 4, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('šedotón')
    plt.axis('off')

    plt.subplot(3, 4, 3)
    plt.imshow(blurred, cmap='gray')
    plt.title('filter')
    plt.axis('off')

    plt.subplot(3, 4, 4)
    plt.imshow(adaptive_thresh, cmap='gray')
    plt.title('prahování')
    plt.axis('off')

    plt.subplot(3, 4, 5)
    plt.imshow(combined_thresh, cmap='gray')
    plt.title('druhý prahování')
    plt.axis('off')

    plt.subplot(3, 4, 6)
    plt.imshow(comb_ero, cmap='gray')
    plt.title('druhý prahování')
    plt.axis('off')

    plt.subplot(3, 4, 7)
    plt.imshow(out , cmap='gray')
    plt.title('odstranění malých objektů')
    plt.axis('off')

    plt.subplot(3, 4, 8)
    plt.imshow(kostra_im, cmap='gray')
    plt.title('kostra')
    plt.axis('off')

    plt.subplot(3, 4, 10)
    plt.imshow(incision, cmap='gray')
    plt.title('kostra')
    plt.axis('off')

    plt.subplot(3, 4, 11)
    plt.imshow(stehy, cmap='gray')
    plt.title('kostra')
    plt.axis('off')

    plt.tight_layout()  # Upraví rozvržení tak, aby se obrázky nepřekrývaly
    plt.show()
    return None
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
    gray = rgb2gray(image)
    blurred = ndimage.gaussian_filter(gray, sigma=1)
    # Aplikace adaptivního prahování s Gaussovým průměrem
    adaptive_thresh = cv2.adaptiveThreshold((blurred* 255).astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 91, 10)

    # První prahování
    _, binary_thresh1 = cv2.threshold(adaptive_thresh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Druhé prahování s jinou hodnotou prahu
    _, binary_thresh2 = cv2.threshold(adaptive_thresh, 200, 255, cv2.THRESH_BINARY)

    # Kombinace prahování
    combined_thresh = cv2.bitwise_or(binary_thresh1, binary_thresh2)
    skimage.io.imsave("../images/threshold/"+im.split(".")[0]+".png", img_as_ubyte(combined_thresh))
    kernel = diamond(1)

    labelled = morphology.label(combined_thresh)
    rp = measure.regionprops(labelled)
    size = [i.area for i in rp]
    size.sort()
    out = morphology.remove_small_objects(combined_thresh.astype(bool), min_size=size[-1]*0.3)

    kostra_im= kostra(out)
    skimage.io.imsave("../images/skeletons/"+im.split(".")[0]+".png", img_as_ubyte(kostra_im))
    stehy,incision = remove_incision(kostra_im)
    labels, num = morphology.label(stehy, return_num=True)
    skimage.io.imsave("../images/incisionless/"+im.split(".")[0]+".png", img_as_ubyte(stehy))
    output[im] = num
    
    write_output("output.csv" , output)
    
    visualize(image,gray,blurred,adaptive_thresh,combined_thresh,combined_thresh,out,kostra_im,incision,stehy)
