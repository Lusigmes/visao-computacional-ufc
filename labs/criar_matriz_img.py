import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def redimensionar_imagem(imagem, largura_nova, altura_nova):
    """
    Redimensiona a imagem para a nova largura e altura.
    """
    return cv2.resize(imagem, (largura_nova, altura_nova))

# Carregar as imagens
imagem1 = mpimg.imread('img/1.png')
imagem2 = mpimg.imread('img/2.png')
imagem3 = mpimg.imread('img/3.png')
imagem4 = mpimg.imread('img/4.png')
imagem5 = mpimg.imread('img/5.png')
imagem6 = mpimg.imread('img/6.png')

# Redimensionar as imagens
largura_nova = 500
altura_nova = 300
imagem1 = redimensionar_imagem(imagem1, largura_nova, altura_nova)
imagem2 = redimensionar_imagem(imagem2, largura_nova, altura_nova)
imagem3 = redimensionar_imagem(imagem3, largura_nova, altura_nova)
imagem4 = redimensionar_imagem(imagem4, largura_nova, altura_nova)
imagem5 = redimensionar_imagem(imagem5, largura_nova, altura_nova)
imagem6 = redimensionar_imagem(imagem6, largura_nova, altura_nova)

# Criar a figura e os eixos
fig, axs = plt.subplots(3, 2, figsize=(10, 15))  # Ajustar o tamanho da figura se necessário

# Adicionar imagens aos eixos
axs[0, 0].imshow(imagem1)
# axs[0, 0].set_title('Etapa 1')
axs[0, 0].axis('off')

axs[0, 1].imshow(imagem2)
# axs[0, 1].set_title('Etapa 2')
axs[0, 1].axis('off')

axs[1, 0].imshow(imagem3)
# axs[1, 0].set_title('Etapa 3')
axs[1, 0].axis('off')

axs[1, 1].imshow(imagem4)
# axs[1, 1].set_title('Etapa 4')
axs[1, 1].axis('off')

axs[2, 0].imshow(imagem5)
# axs[2, 0].set_title('Etapa 5')
axs[2, 0].axis('off')

axs[2, 1].imshow(imagem6)
# axs[2, 1].set_title('Etapa 6')
axs[2, 1].axis('off')

# Ajustar o layout
plt.tight_layout()

# Exibir a matriz de imagens
plt.show()
