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
    # Načtení obrázku
    image = skimage.io.imread("../images/incision_couples/" + im)
    
    # Konverze do šedotónu
    gray = rgb2gray(image)
    
    # Aplikace adaptivního prahování s Gaussovým průměrem
    blurred = ndimage.gaussian_filter(gray, sigma=1)
    adaptive_thresh = cv2.adaptiveThreshold((blurred * 255).astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 91, 10)

    # Detekce čar pomocí Houghovy transformace
    lines = cv2.HoughLinesP(adaptive_thresh, 1, np.pi/180, threshold=20, minLineLength=30, maxLineGap=5)
    lines_binary = np.zeros_like(gray)
    
    # Vykreslení nalezených linií
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length > 50:  # Filtrování malých segmentů
                cv2.line(lines_binary, (x1, y1), (x2, y2), (255), 2)  # Vykreslení linie bílou barvou
    kernel = np.ones((3, 3), np.uint8)
    lines_binary = cv2.dilate(lines_binary, kernel, 1)
    # Detekce hran pouze z levé strany s větší intenzitou
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_combined = np.sqrt(sobel_x**2)
    edge_mask = np.zeros_like(gray, dtype=np.uint8)
    edge_mask[(sobel_x > 0) & (sobel_combined > 115)] = 255
    
    # Výpočet rozdílu mezi hranami a liniemi
    result_image = np.subtract(edge_mask, lines_binary)
    result_image[result_image > 0] = 255
    result_image[result_image <= 0] = 0
    


    
    # Dilatace pro spojení případných oddělených segmentů
    kernel = np.ones((100, 4), np.uint8)
    dilated_image = cv2.dilate(result_image, kernel, 1)


    # Identifikace jednotlivých objektů v obrázku
    labeled_image, num_features = ndimage.label(dilated_image)

    # Pokud je obrázek prázdný (neobsahuje žádné objekty), vrátíme prázdný obrázek
    if num_features == 0:
        print("Obrázek je prázdný. Nejsou zde žádné objekty k detekci.")
        pass

    # Získání centroidů jednotlivých objektů
    centroids = ndimage.center_of_mass(dilated_image, labeled_image, range(1, num_features + 1))

    # Seřazení centroidů podle jejich polohy na ose x
    centroids = sorted(centroids, key=lambda x: x[1])

    # Rozmezí pro očekávaný počet čar
    expected_lines_range = range(5, 7)

    # Vypočet rozestupů mezi centroidy
    gaps = [centroids[i+1][1] - centroids[i][1] for i in range(len(centroids)-1) if i < len(centroids)-1]

    # Průměrný rozestup mezi centroidy
    avg_gap = np.mean(gaps) if gaps else 0

    # Tolerance pro rozestupy
    tolerance = 0.2 * avg_gap

    # Detekce počtu čar
    detected_lines = len(centroids)

    # Pokud je počet nalezených čar 5 nebo méně, použijeme všechny nalezené čáry
    if num_features <= 5:
        cleaned_centroids = centroids
    else:
        # Vypočet rozestupů mezi centroidy
        gaps = [centroids[i+1][1] - centroids[i][1] for i in range(len(centroids)-1) if i < len(centroids)-1]

        # Průměrný rozestup mezi centroidy
        avg_gap = np.mean(gaps) if gaps else 0

        # Tolerance pro rozestupy
        tolerance = 0.2 * avg_gap

        # Detekce počtu čar
        detected_lines = len(centroids)

        # Nalezení očekávaného počtu čar v rozmezí
        expected_lines = min(range(5, 7), key=lambda x: abs(x - detected_lines))

        # Odstranění chybných centroidů na základě rozestupů
        cleaned_centroids = [centroids[0]]
        for i in range(1, len(centroids)):
            if len(cleaned_centroids) < expected_lines and np.abs(centroids[i][1] - cleaned_centroids[-1][1]) >= avg_gap - tolerance:
                cleaned_centroids.append(centroids[i])

    # Vytvoření prázdného obrázku pro výsledek
    cleaned_image = np.zeros_like(dilated_image)

    # Vykreslení centroidů na výsledečný obrázek
    for centroid in cleaned_centroids:
        cv2.circle(cleaned_image, (int(centroid[1]), int(centroid[0])), 3, 255, -1)

    
    # Zobrazení obrázků
    plt.figure(figsize=(12, 8))
    
    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Původní obrázek')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow( edge_mask, cmap='gray')
    plt.title('Detekované hrany')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow( dilated_image, cmap='gray')
    plt.title('Dilatace')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(cleaned_image, cmap='gray')
    plt.title('celkový počet')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
