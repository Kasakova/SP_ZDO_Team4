
import numpy as np
from skimage import measure, morphology,img_as_ubyte


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
    img_copy = np.copy(im)
    labelled = morphology.label(img_copy)
    rp = measure.regionprops(labelled)
    incision = np.zeros_like(img_copy)
    for i in rp:
        y1,x1,y2,x2 = i.bbox#(min_row, min_col, max_row, max_col)
        yy, xx= np.nonzero(img_copy[y1:y2,x1:x2])
        yy = yy+y1
        xx = xx+x1
        try:
            index_vpravo = np.argmax(xx) #pravy bod
            index_vlevo = np.argmin(xx) #levy bod
        except ValueError:
            return img_copy #prazdny obraz, nebude se nic delat

        inc = find_path(img_as_ubyte(img_copy), (yy[index_vlevo],xx[index_vlevo]), (yy[index_vpravo],xx[index_vpravo]))
        incision[inc] = 1

    img_copy[incision] = 0

    kernel = np.ones((5,5))
    # kernel = morphology.diamond(1)

    img_copy = morphology.binary_dilation(img_copy, kernel)

    labelled = morphology.label(img_copy)
    rp = measure.regionprops(labelled)

    size = [i.area for i in rp]
    size.sort()
    if len(size)!=0:
        out = morphology.remove_small_objects(img_copy.astype(bool), min_size=size[-1]*0.5)
        out = morphology.remove_small_objects(img_copy.astype(bool), min_size=img_copy.shape[0]*0.3*kernel.shape[0])# TODO nejak odstranit naopak velky objekty?
    else:
        out = img_copy
    return out,incision






