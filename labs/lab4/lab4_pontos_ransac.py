# ler imagem para traçar a reta (ta quebrando)
import cv2
import matplotlib.pyplot as plt
import numpy as np
from random import random, randint

path = "imgs/pontos_ransac.png"  
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

bordas = cv2.Canny(gray, 50, 150)

# coordenadas dos pontos das bordas
points = np.column_stack(np.where(bordas > 0))

# Converte os pontos para uma lista de tuplas (x, y)
allPoints = [(point[1], point[0]) for point in points]

# gerar um par de índices aleatórios
def randomIndicesPair():
    index1 = randint(0, len(allPoints)-1)
    index2 = randint(0, len(allPoints)-1)
    while index1 == index2:
        index2 = randint(0, len(allPoints)-1)
    return (index1, index2)

# Ajustar linha usando RANSAC
totalIterations = 100
tolerance = 1
max_inliersCounts = 0
coeficients = []

for iteration in range(totalIterations):
    (index1, index2) = randomIndicesPair()
    A = allPoints[index1][1] - allPoints[index2][1]
    B = allPoints[index2][0] - allPoints[index1][0]
    C = (allPoints[index1][0] * allPoints[index2][1]) - (allPoints[index1][1] * allPoints[index2][0])
    
    # equação (y1 -y2)*X + (x2 - x1)*Y + (x1*y2 - y1*x2) = 0

    countInliers = 0
    for i in range(len(allPoints)):
        distance = abs(A * allPoints[i][0] + B * allPoints[i][1] + C) / (pow(A * A + B * B, 0.5))
        if distance < tolerance:
            countInliers += 1
    if countInliers > max_inliersCounts:
        coeficients = [A, B, C]
        max_inliersCounts = countInliers

# inha ajustada na imagem
A = coeficients[0]
B = coeficients[1]
C = coeficients[2]

allXvalues = [point[0] for point in allPoints]
xValues = [min(allXvalues), max(allXvalues)]
yValues = [int((-C - A * xValues[0]) / B), int((-C - A * xValues[1]) / B)]

# Desenhar
cv2.line(img, (xValues[0], yValues[0]), (xValues[1], yValues[1]), (0, 0, 255), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Line Through Points")
plt.axis('off')
plt.show()

print(f"Inliers: {max_inliersCounts}")
