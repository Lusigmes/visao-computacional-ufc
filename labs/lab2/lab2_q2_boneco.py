import cv2
import numpy as np


# Criar uma imagem branca (300x300 pixels, 3 canais de cor para RGB)
fundo = np.ones((300, 300, 3), dtype=np.uint8) * 255

################################# #cabeça ########################################

circle = cv2.imread("imgs/circle.jpg")
# circle = cv2.resize(circle, (100, 100))  # Redimensiona o círculo para 100x100 pixels

x_circle = 100
y_circle= 0

# Inserir a imagem do círculo no fundo
fundo[y_circle:y_circle+circle.shape[0], x_circle:x_circle+circle.shape[1]] = circle

################################# tronco ########################################

line = cv2.imread("imgs/line.jpg")

#dobrar
w, h = line.shape[1], line.shape[0]
ww = w+w//2
line_dobrarda = cv2.resize(line, (ww, h))

#vertical
line_vertical = cv2.transpose(line_dobrarda)
line_vertical = cv2.flip(line_vertical, flipCode=1)

#
x_tronco = 100
y_tronco = 71
fundo[y_tronco:y_tronco+line_vertical.shape[0], x_tronco:x_tronco+line_vertical.shape[1]] = line_vertical

################################# #braços ########################################

braco_w = int(0.75 * line_vertical.shape[1])
braco_h = int(0.75 * line.shape[0])

# dirieto
braço_dir = line
#
braço_dir_redimensionado = cv2.resize(braço_dir, (braco_w, braco_h))
#
x_braco_dir = 55 + line_vertical.shape[1]
y_braco_dir = 70
#
fundo[y_braco_dir:y_braco_dir+braço_dir_redimensionado.shape[0], x_braco_dir:x_braco_dir+braço_dir_redimensionado.shape[1]] = braço_dir_redimensionado

# esquerto
braço_esq = line
#
# Redimensionar o braço esquerdo para 75% do tamanho do tronco
braço_esq_redimensionado = cv2.resize(braço_esq, (braco_w, braco_h))

# Posicionar o braço esquerdo
x_braco_esq = 170 - line_vertical.shape[1]  
y_braco_esq= 70 # mesma altura do outro braço
#
fundo[y_braco_esq:y_braco_esq+braço_esq_redimensionado.shape[0], x_braco_esq:x_braco_esq+braço_esq_redimensionado.shape[1]] = braço_esq_redimensionado

################################# #pernas ########################################
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return rotated


perna_w = 2 * braco_w
perna_h = braco_h
#
#direita
#
perna_dir = cv2.resize(braço_dir, (perna_w, perna_h))
perna_dir_rot = rotate_image(perna_dir, 45)  # Rotaciona a perna direita a 45º
#

x_perna_dir = 90 - perna_w // 2
y_perna_dir = 210
#
fundo[y_perna_dir:y_perna_dir+perna_h, x_perna_dir:x_perna_dir+perna_w] = perna_dir_rot

# Esquerda
perna_esq = cv2.resize(braço_esq, (perna_w, perna_h))
perna_esq_rot = rotate_image(perna_esq, -45)  # Rotaciona a perna esquerda a -45º

x_perna_esq = 60 + perna_w //2
y_perna_esq = 210
fundo[y_perna_esq:y_perna_esq+perna_h, x_perna_esq:x_perna_esq+perna_w] = perna_esq_rot

# Exibir a imagem
cv2.imshow('Imagem Criada', fundo)
# cv2.imshow('circle', line)
# cv2.imshow('line', line_vertical)
cv2.waitKey(0)
cv2.destroyAllWindows()
