import cv2
import numpy as np

# aplicar correção gama para tornar imagem sépia

image = cv2.imread("imgs/jato.jpg")

gamma = 4.5
tabela = np.empty((1, 256), np.uint8)

for i in range(256):
    tabela[0, i] = np.clip(pow(i/ 255.0, gamma) * 255.0, 0, 255)
    
gamma_ = cv2.LUT(image, tabela)

#canais de cores
b, g, r = cv2.split(gamma_)
r = cv2.add(r, 122)
g = cv2.add(g, 75)
b = cv2.add(b, 15)

sepia = cv2.merge((b, g, r))

cv2.imshow("original", image)
cv2.imshow("sepia", sepia)

cv2.waitKey(0)
cv2.destroyAllWindows()