# cria matriz com pontos

import matplotlib.pyplot as plt
from random import random, randint

# gerar pontos
inliersTolerance = 2
inliersCount = 20
inliers = [(i +  random()*inliersTolerance, 2 * (i + random()*inliersTolerance))
    for i in range(0, inliersCount)]

outliersCount = 20
outliers = [(random()*outliersCount, 2 * (random()*outliersCount))
    for i in range(0, outliersCount)]

allPoints = inliers + outliers
for point in allPoints:
    plt.scatter(point[0], point[1], color = "blue")
plt.title("points 2D")
plt.show()

# gerar par de indices random
def randomIndicesPair():
    index1 = randint(0, len(allPoints)-1)
    index2 = randint(0, len(allPoints)-1)
    while index1 == index2:
        index2 = randint(0, len(allPoints)-1)

    return (index1, index2)

# traçar pontos
totalInterations = 100
tolerance = 1
max_inliersCounts = 0
coeficients = []

# equação (y1 -y2)*X + (x2 - x1)*Y + (x1*y2 - y1*x2) = 0
for interation in range(0, totalInterations):
    (index1, index2) = randomIndicesPair()
    
    A = allPoints[index1][1] - allPoints[index2][1]        
    B = allPoints[index2][0] - allPoints[index1][0]        
    C = (allPoints[index1][0]*allPoints[index2][1]) - (allPoints[index1][1]*allPoints[index2][0])        

    countInliers = 0
    for i in range(0, len(allPoints)):
        # distancia perpendicular dos pontos x1, y1
        # na linha Ax + By + C = 0 is |Ax1 + By1 + C|/sqrt(A^2 + B^2)
        distance = abs(A*allPoints[i][0] + B*allPoints[i][1] + C)/(pow(A*A + B*B, 0.5))
        if distance < tolerance:
            countInliers += 1
    if countInliers > max_inliersCounts:
        coeficients = [A, B, C]
        max_inliersCounts = countInliers

for point in allPoints:
    plt.scatter(point[0], point[1], color= "green")

A = coeficients[0]
B = coeficients[1]
C = coeficients[2]

allXvalues = [point[0] for point in allPoints]
xValues = [min(allXvalues), max(allXvalues)]
yValues = [(-C-A*xValues[0])/B, (-C-A*xValues[1])/B]
#Y = -C-A*X/B

plt.plot(xValues, yValues, color="red")
plt.title("line through points")
plt.show()

print("inliers: {0}".format(max_inliersCounts))
