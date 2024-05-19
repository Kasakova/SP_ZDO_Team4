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
def detect_and_plot_circles(image_path):
    # Načtení obrázku
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detekce kruhů pomocí transformace Houghova kruhu s upravenými parametry pro hledání menších kruhů
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=5, param1=20, param2=15, minRadius=2, maxRadius=10)

    if circles is not None:
        # Zaokrouhlení souřadnic a poloměrů kruhů
        circles = np.round(circles[0, :]).astype("int")

        # Nakreslení nalezených kruhů na originální obrázek
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 1)

        # Vytvoření figure a subplotů
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Detekce kruhů')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(gray, cmap='gray')
        plt.title('Šedotón')
        plt.axis('off')



        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 1)
        plt.title('Detekované kruhy')
        plt.axis('off')

        plt.tight_layout()
        plt.show()
    else:
        print("No circles detected.")


def list2dict(lines):
    dict = {}
    for line in lines:
        spl = line.split(",")
        if spl[0] != "filename":
            dict[spl[0]] = spl[1]
    return dict



# Hlavní část kódu
images = os.listdir("../images/incision_couples/")
output = {}
a= 0
b=0
for im in images:
        # Konverze do šedotónu
    image = skimage.io.imread("../images/incision_couples/" + im)
    
    # Převést obrázek na formát s plovoucí desetinnou čárkou
    obrazek_float = np.float32(image)
    
    # Rozdělit obrázek na jednotlivé kanály (BGR)
    kanaly = cv2.split(obrazek_float)
    kanaly = [kanal * 1. for kanal in kanaly]
    
    # Sloučit upravené kanály zpět do obrázku
    vysledek = cv2.merge(kanaly)
    
    # Omezit hodnoty na rozsah 0-255
    vysledek = np.clip(vysledek, 0, 255)
    
    # Převést zpět na datový typ uint8 (8-bitový bez znaménka)
    vysledek = np.uint8(vysledek)
    
    # Konverze do šedotónu
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Výpočet elevační mapy
    elevation_map = sobel(gray)

    # Inicializace markerů
    markers = np.zeros_like(gray)

    # Výpočet prahu Otsu
    thresh = threshold_otsu(np.asarray(gray))

    # Nastavení markerů pro tmavé objekty
    markers[gray < thresh-10] = 1
    markers[gray > thresh] = 2

    # Segmentace pomocí algoritmu vodního rozsedu
    segmentation = skimage.segmentation.watershed(elevation_map, markers)


    
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Detekce čar pomocí Houghovy transformace
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
    
    lines = cv2.HoughLinesP(edges , 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

    # Vytvoření kopie obrázku pro vykreslení nalezených čar
    lines_image = np.copy(image)
    csv_file = 'anotace.csv'
    # Projdeme nalezené čáry a vykreslíme je do kopie obrázku
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
                a=a+1

        print(im)
        
        plt.imshow(image)
        plt.tight_layout()
        #plt.show()
        
print(a)
