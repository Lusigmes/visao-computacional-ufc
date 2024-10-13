import numpy as np
import cv2

image = cv2.imread("gamora_nebula.jpg")

imager = cv2.resize(image, (600,500), interpolation=cv2.INTER_AREA)

img_hsv = cv2.cvtColor(imager, cv2.COLOR_BGR2HSV)

def rgb_to_hsv(rgb):
    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]

#############################################################################

# # Gamora
# alguns tons da pele identificado
cores_rgb_gamora = [
    (35, 70, 50),
    (75, 255, 150),
    (221, 207, 106),
    (245, 220, 30),
    (248, 220, 149),
    (231, 225, 185)
]

# Converte as cores RGB para HSV
cores_hsv_g = [rgb_to_hsv(c) for c in cores_rgb_gamora]
# limites inf sup verde
lower_g = np.min(cores_hsv_g, axis=0)
upper_g = np.max(cores_hsv_g, axis=0)
# mascarar
mask_g = cv2.inRange(img_hsv, lower_g, upper_g)
#
orange = np.zeros_like(imager, np.uint8)
orange[:] = [0, 165, 255] 
# bitiuaise
result_g = cv2.bitwise_and(orange, orange, mask=mask_g)
mask_invertida_g = cv2.bitwise_not(mask_g)
bg_original_g = cv2.bitwise_and(imager, imager, mask=mask_invertida_g)
finall_g = cv2.add(bg_original_g, result_g)
####


#  Nebula
cores_rgb_nebula = [
    (4, 29, 46),
    (8, 99, 140),
    (5, 34, 50),
    (2, 50, 73),
    (127, 204, 235),
    (3, 160, 211),
    (108, 182, 228),
    (45, 115, 174),
    (175, 226, 248),
    (2, 176, 210),
    (8, 55, 99),                                                                                                                
    (2, 25, 40),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    (188, 229, 242),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
]
cores_hsv_n = [rgb_to_hsv(c) for c in cores_rgb_nebula]
# limites inf sup verde
lower_n = np.min(cores_hsv_n, axis=0)
upper_n = np.max(cores_hsv_n, axis=0)
# mascarar
mask_n = cv2.inRange(img_hsv, lower_n, upper_n)
#
white = np.zeros_like(imager, np.uint8)
white[:] = [255, 255, 255] 
# bitiuaise
result_n = cv2.bitwise_and(white, white, mask=mask_n)
mask_invertida_n = cv2.bitwise_not(mask_n)
bg_original_n = cv2.bitwise_and(imager, imager, mask=mask_invertida_n)
finall_n = cv2.add(bg_original_n, result_n)
################################################################################

# juntar td
finall = cv2.add(finall_g, finall_n)

# Exibir os resultados
cv2.imshow("Resultado", finall)
cv2.imshow("gamora", finall_g)
cv2.imshow("nebula", finall_n)

cv2.waitKey(0)
cv2.destroyAllWindows()