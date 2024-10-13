import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import io, transform
from skimage import filters
from skimage import color
import time

#############################################################################

def calculate_energy(image): #  Determina a importância de cada pixel
    gray = color.rgb2gray(image)
    energy = np.abs(filters.sobel(gray)) #sobel_v ? #gradiente horizontal da imagem
    return energy

#############################################################################

def find_seam(energy): # Identifica o caminho de menor energia através da imagem
    row, col = energy.shape  # row, column
    M = energy.copy() # matriz de energia acumulada
    backtrack = np.zeros_like(M, dtype=int)
    #preencher matriz de energia acumulada
    for i in range(1, row):
        for j in range(col):
            #bordas tratadas separadamente
            if j == 0:
                index = np.argmin( M[i-1, j:j+2])
                backtrack[i, j] = index + j
                min_energy = M[i-1, index+j]
            else:
                index = np.argmin( M[i-1, j-1:j+2])
                backtrack[i, j] = index + j-1
                min_energy = M[i-1, index + j-1]
            M[i, j] += min_energy
    return M, backtrack

def remove_seam(image, backtrack): # Remove o seam identificado, reduzindo a largura da imagem
    row, col, _ = image.shape
    output = np.zeros((row, col - 1, 3), dtype = image.dtype)
    j = np.argmin(backtrack[-1])
    for i in reversed(range(row)):
        output[i, :, 0] = np.delete(image[i, :, 0], [j])
        output[i, :, 1] = np.delete(image[i, :, 1], [j])
        output[i, :, 2] = np.delete(image[i, :, 2], [j])
        j = backtrack[i, j]

    return output

#############################################################################

def seam_carving(image, num_seans, direction='vertical', max_time = 300):
    start_time = time.time()
    for _ in range(num_seans):
        if time.time() - start_time > max_time:
            print("Tempo Máximo Excedido")
            break
        
        if direction == 'horizontal': #Reduzir a altura da imagem. (rotaciona para tratar colunas como linhas)
            image = np.rot90(image, 1, (0, 1))

        energy = calculate_energy(image=image)
        M, backtrack = find_seam(energy=energy)
        image = remove_seam(image=image, backtrack=backtrack)
 
        if direction == 'horizontal': 
            image = np.rot90(image, -1, (0, 1)) #Reduzir a largura da imagem. volta para vertical (des-rotaciona)

    return image

#############################################################################

def remove_object(image, mask):
    mask = np.where(mask > 0, 1, 0).astype(np.uint8)
    seam_img = image.copy()

    while np.any(mask):
        energy = calculate_energy(seam_img)
        energy += mask * 1e6

        M, backtrack = find_seam(energy=energy)
        seam_img = remove_seam(seam_img, backtrack)
        mask = np.delete(mask, np.argmin(M[-1]), axis=1)

    return seam_img

#############################################################################
 
img = io.imread('images/balls.jpg')
img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))

print("Computando...")

new_img_horizontal = seam_carving(img, 40, direction='horizontal') 
new_img_vertical = seam_carving(img, 40, direction='vertical') 

print("Computando[1]...")

mask = np.zeros(img.shape[:2], dtype=np.uint8)
mask[50:100, 100:200] = 255

print("Finalizando...")

new_img_removed = remove_object(img, mask)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))  # Ajuste para 2x2 e altere o figsize conforme necessário

ax[0, 0].imshow(img)
ax[0, 0].set_title("Original")
ax[0, 0].axis('off')

ax[0, 1].imshow(new_img_horizontal)
ax[0, 1].set_title("Seam Carved Horizontal")
ax[0, 1].axis('off')

ax[1, 0].imshow(new_img_vertical)
ax[1, 0].set_title("Seam Carved Vertical")
ax[1, 0].axis('off')

ax[1, 1].imshow(new_img_removed)
ax[1, 1].set_title("Seam Carved Removed Object")
ax[1, 1].axis('off')

plt.tight_layout()  # Ajusta os subplots para evitar sobreposição
plt.show()


plt.tight_layout()
plt.savefig("seam_carved - complete")
# plt.savefig("seam_carved")
print("Encerrado...")