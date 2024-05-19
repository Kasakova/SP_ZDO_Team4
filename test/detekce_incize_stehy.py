import cv2
import numpy as np
from skimage.color import rgb2gray
from scipy import ndimage

def detekce(image):
    potential_stitch = 0
    # Konverze do šedotónu
    gray = rgb2gray(image)
    
    # Aplikace adaptivního prahování s Gaussovým průměrem
    blurred = ndimage.gaussian_filter(gray, sigma=1)
    adaptive_thresh = cv2.adaptiveThreshold((blurred * 255).astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 91, 10)

    # Detekce čar pomocí Houghovy transformace
    lines = cv2.HoughLinesP(adaptive_thresh, 1, np.pi / 180, threshold=20, minLineLength=30, maxLineGap=5)
    lines_binary = np.zeros_like(gray)
    
    # Vykreslení nalezených linií
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > 50:  # Filtrování malých segmentů
                cv2.line(lines_binary, (x1, y1), (x2, y2), (255), 2)  # Vykreslení linie bílou barvou
    else:
        potential_stitch = -1
        return potential_stitch

    kernel = np.ones((3, 5), np.uint8)
    lines_binary = cv2.dilate(lines_binary, kernel, 1)
    
    # Detekce hran pouze z levé strany s větší intenzitou
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_combined = np.sqrt(sobel_x ** 2)
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

    # Získání centroidů jednotlivých objektů
    centroids = ndimage.center_of_mass(dilated_image, labeled_image, range(1, num_features + 1))

    # Seřazení centroidů podle jejich polohy na ose x
    centroids = sorted(centroids, key=lambda x: x[1])

    # Pokud je počet nalezených čar 5 nebo méně, použijeme všechny nalezené čáry
    if num_features <= 5:
        cleaned_centroids = centroids
    else:
        # Vypočet rozestupů mezi centroidy
        gaps = [centroids[i + 1][1] - centroids[i][1] for i in range(len(centroids) - 1) if i < len(centroids) - 1]

        # Průměrný rozestup mezi centroidy
        avg_gap = np.mean(gaps) if gaps else 0

        # Tolerance pro rozestupy
        tolerance = 0.2 * avg_gap

        # Detekce počtu čar
        detected_lines = len(centroids)

        # Odstranění chybných centroidů na základě rozestupů
        cleaned_centroids = [centroids[0]]
        for i in range(1, len(centroids)):
            if len(cleaned_centroids) < 5 and np.abs(centroids[i][1] - cleaned_centroids[-1][1]) >= avg_gap - tolerance:
                cleaned_centroids.append(centroids[i])
            if len(cleaned_centroids) >= 5:
                break

    potential_stitch = len(cleaned_centroids)
    
    if potential_stitch < 5:
        potential_stitch = -1
         
    return potential_stitch
    

    
