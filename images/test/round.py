import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Zavolej funkci s cestou k obrázku
detect_and_plot_circles("../../images/incision_couples/SA_20220620-103036_3vhpgd00n9vt_incision_crop_0_start.jpg")