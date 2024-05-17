import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import canny
from scipy import ndimage
import numpy as np
from skimage.measure import label, regionprops
import skimage.io
import os

def visualize(image, gray, edges, fill_edges, filtered_image, closed_image):
    # Zobrazení výsledků
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title('Původní obrázek')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Šedotón')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(edges, cmap='gray')
    plt.title('Detekce hran')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(fill_edges, cmap='gray')
    plt.title('Vyplnění')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Filtrování')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(closed_image, cmap='gray')
    plt.title('Uzavření')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def write_output(file, output):
    with open(file, "w", encoding="UTF8") as f:
        f.write("filename, n_stitches\n")
        for item in output:
            f.write(f"{item}, {output[item]}\n")

def count_unique_objects(binary_image):
    # Označení objektů v obrázku
    labeled_image, num_objects = ndimage.label(binary_image)
    
    # Seznam pro ukládání souřadnic centroidů objektů
    centroids = []
    
    # Prahová hodnota pro porovnávání souřadnic x
    x_threshold = 10  # Můžete upravit podle potřeby
    
    # Procházení objektů a jejich centroidů
    for region in regionprops(labeled_image):
        centroids.append(region.centroid)
    
    # Set pro ukládání jedinečných objektů na základě souřadnic x
    unique_objects = set()
    
    # Přidání objektů, pokud jejich souřadnice x nejsou podobné jinému objektu
    for c in centroids:
        if not any(abs(c[1] - u[1]) < x_threshold for u in unique_objects):
            unique_objects.add(c)
    
    # Počet jedinečných objektů
    unique_count = len(unique_objects)
    
    return unique_count

# Hlavní část kódu
images = os.listdir("../images/incision_couples/")
output = {}

for im in images:
    # Konverze do šedotónu
    image = skimage.io.imread("../images/incision_couples/" + im)
    gray = rgb2gray(image)
   
    edges = canny(gray)
    structure = np.ones((4, 4), dtype=np.uint8)
    edges_a = ndimage.binary_closing(edges, structure=structure)
   
    fill_edges = ndimage.binary_fill_holes(edges_a)
    
    label_image = ndimage.label(fill_edges)[0]
        
    # Filtrace objektů podle kulatosti
    filtered_image = np.zeros_like(label_image)
    for region in regionprops(label_image):
        # Filtrace podle kulatosti
        eccentricity = region.eccentricity
        if eccentricity < 0.85:  # Pragmatická hodnota pro kulatost, můžete upravit podle potřeby
            for coord in region.coords:
                filtered_image[coord[0], coord[1]] = 255
    
    # Definice strukturovaného prvku
    structure = np.ones((3, 3), dtype=np.uint8)  # 3x3 strukturovaný prvek, můžete upravit podle potřeby

    # Provádění morfologického uzavření (closing)
    closed_image = ndimage.binary_opening(filtered_image, structure=structure)
    
    # Označení uzavřeného obrázku
    closed_image_label = ndimage.label(closed_image)[0]
    
    # Počet jedinečných objektů
    unique_count = count_unique_objects(closed_image)
    output[im] = 5-unique_count
    
    print(f"{im}: {5-unique_count}")
    
    write_output("output.csv", output)
    #visualize(image, gray, edges, fill_edges, filtered_image, closed_image)