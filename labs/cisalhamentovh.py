import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread("jato.jpg")
# img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

rows, cols, _ = image.shape

shx=0.3
shy=0.2

new_cols = int(cols + abs(shx) * rows)
new_rows = int(rows + abs(shy) * cols)

m_h = np.float32([[1, shx, 0], [0,1,0]])
shared_h = cv2.warpAffine(image, m_h, (new_cols, rows))

m_v = np.float32([[1, 0, 0], [shy,1,0]])
shared_v = cv2.warpAffine(image, m_v, (cols, new_rows))


# Exibe a imagem original
plt.figure()  # Nova janela
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Converte de BGR para RGB para exibir corretamente no Matplotlib
plt.title("Imagem Original")
plt.axis('off')  # Remove os eixos

# Exibe a imagem com cisalhamento vertical
plt.figure()  # Nova janela
plt.imshow(cv2.cvtColor(shared_v, cv2.COLOR_BGR2RGB))  # Converte de BGR para RGB
plt.title("Imagem Cisalhada Verticalmente")
plt.axis('off')

# Exibe a imagem com cisalhamento horizontal
plt.figure()  # Nova janela
plt.imshow(cv2.cvtColor(shared_h, cv2.COLOR_BGR2RGB))  # Converte de BGR para RGB
plt.title("Imagem Cisalhada Horizontalmente")
plt.axis('off')

plt.show()