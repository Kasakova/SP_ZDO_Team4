import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology
from skimage.filters import sobel
from skimage.filters import try_all_threshold,threshold_otsu,threshold_yen,threshold_local
import os
import csv
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage import io, morphology
from scipy import ndimage



# Hlavní část kódu
images = os.listdir("../images/incision_couples/")
output = {}
a= 0
b=0
seznam = ['SA_20220620-102836_e8o0aggtd2us_incision_crop_0_start.jpg', 
          'SA_20220707-190320_tl4ev95290pp_incision_crop_0_start.jpg', 
          'SA_20230221-195759_f7pcqna517us_incision_crop_0_start.jpg',
          'SA_20230221-205917_nu2i64xl9cio_incision_crop_0_start.jpg',
          'SA_20231011-104231_e3sdqld32ut6_incision_crop_0_start.jpg',
          'SA_20240213-102404_ehnxsn46xjb3_incision_crop_0_start.jpg',
          'SA_20240215-113735_s7d6x829ttl3_incision_crop_0_start.jpg',
          'SA_20240220-121134_uwke7gr7u8r0_incision_crop_0_start.jpg',
          'SA_20240223-234821_5mnv5n0fxzi8_incision_crop_0.jpg',
          'SA_20240223-234821_5mnv5n0fxzi8_incision_crop_0_start.jpg']
for im in images:
        # Konverze do šedotónu
    image = skimage.io.imread("../images/incision_couples/" + im)



    gray = rgb2gray(image)


    blurred = ndimage.gaussian_filter(gray, sigma=1)
    # Aplikace adaptivního prahování s Gaussovým průměrem
    adaptive_thresh = cv2.adaptiveThreshold((blurred* 255).astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 91, 10)

    # Detekce čar pomocí Houghovy transformace
    lines = cv2.HoughLinesP(adaptive_thresh, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

        # Vytvoření kopie obrázku pro vykreslení nalezených čar
    lines_image = np.copy(image)

    # Filtrování vodorovných čar
    horizontal_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        plt.imshow(lines_image)
        plt.tight_layout()
        #plt.show()
    else:
        for jmeno in seznam:
            if jmeno == im:
                print("dobře")
                a += 1

        print(im)
        plt.imshow(image)
        plt.tight_layout()
        plt.show()

print(a)