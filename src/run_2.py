from skimage.color import rgb2gray
import argparse
from skimage.feature import canny
from scipy import ndimage
import numpy as np
from skimage.measure import  regionprops
import skimage.io
import os
import cv2
from detekce_incize_stehy import detekce

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
    x_threshold = 15 
    
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

def process(images):
    output = {}
    potential_stitch=5
    for im in images:
        # Konverze do šedotónu
        image = skimage.io.imread("../images/incision_couples/" + im)
        
        gray = rgb2gray(image)
        blurred = ndimage.gaussian_filter(gray, sigma=1)
        # Aplikace adaptivního prahování s Gaussovým průměrem
        adaptive_thresh = cv2.adaptiveThreshold((blurred* 255).astype('uint8'), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 91, 10)

        # Detekce čar pomocí Houghovy transformace
        lines = cv2.HoughLinesP(adaptive_thresh, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=15)

        if lines is  None:
            output[im] = -1
        else:
            potential_stitch= detekce(image)
            if potential_stitch == -1:
                output[im] = -1
            else:
                edges = canny(gray)
                structure = np.ones((4, 5), dtype=np.uint8)
                edges_a = ndimage.binary_closing(edges , structure=structure)
            
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
                
                # Počet jedinečných objektů
                unique_count = count_unique_objects(closed_image)
                count = 5-unique_count
                if count <0:
                    count =0
                output[im] = count
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
    output = process(os.listdir("../images/incision_couples/"))#vsechny obrazky

    write_output("output.csv", output)
