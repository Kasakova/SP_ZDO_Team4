import cv2
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
from skimage import measure, morphology,img_as_ubyte
from skimage.feature import canny
from scipy import ndimage
import scipy.signal
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

def visualize(image,gray,blurred,edges,edges_dil,fill_edges,out,kostra_im,incision,stehy):
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
    plt.imshow(edges, cmap='gray')
    plt.title('hrany')
    plt.axis('off')

    plt.subplot(3, 4, 5)
    plt.imshow(edges_dil, cmap='gray')
    plt.title('diletace')
    plt.axis('off')

    plt.subplot(3, 4, 6)
    plt.imshow(fill_edges, cmap='gray')
    plt.title('zaplnění')
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
    blurred = ndimage.gaussian_filter(gray, sigma=0)
   
    edges = canny(blurred)
   
    skimage.io.imsave("../images/threshold/"+im.split(".")[0]+".png", img_as_ubyte(edges))
    kernel = diamond(1)
    edges_dil = morphology.binary_dilation(edges, kernel)

    fill_edges = ndimage.binary_fill_holes(edges_dil)

    labelled = morphology.label(fill_edges)
    rp = measure.regionprops(labelled)
    size = [i.area for i in rp]
    size.sort()
    try:
        out = morphology.remove_small_objects(fill_edges, min_size=size[-1]*0.8)
    except IndexError:
        out = fill_edges
    kostra_im= kostra(out)    
    skimage.io.imsave("../images/skeletons/"+im.split(".")[0]+".png", img_as_ubyte(kostra_im))
    stehy,incision = remove_incision(kostra_im)
    labels, num = morphology.label(stehy, return_num=True)
    skimage.io.imsave("../images/incisionless/"+im.split(".")[0]+".png", img_as_ubyte(stehy))
    output[im] = num
  

   



    # Aplikace adaptivního prahování s Gaussovým průměrem
  
 

  

    write_output("output.csv" , output)
    
    visualize(image,gray,blurred,edges,edges_dil,fill_edges,out,kostra_im,incision,stehy)