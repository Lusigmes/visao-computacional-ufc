import matplotlib.pyplot as plt
import cv2
import numpy as np

def detect_corner(path):
    img = cv2.imread(path)

    if img is None:
        print("no image!")
        return
    
    image = np.copy(img) # copia
    print(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    gray = np.float32(gray_image) # imagem cinza para float32

    harris_corner = cv2.cornerHarris(src=gray, blockSize= 5, ksize=3, k=0.02) # aplica detector harris corner

    image[harris_corner > 0.01 * harris_corner.max()] = [0, 0, 0]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("detect corner")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    detect_corner('imgs/img.jpeg')