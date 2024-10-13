from __future__ import print_function
import numpy as np
import argparse
import cv2

#ajusta o gamma de uma imagem de 0.1 a 8.0

def aplicar_gamma(img, gamma=1.0):
    gamma_ = 1.0/ gamma
    tabela = np.array([((i/255.0) ** gamma_) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    
    return cv2.LUT(img, tabela)


argp = argparse.ArgumentParser()
argp.add_argument("-i",  "--image", required=True, help="path to input image")
args = vars(argp.parse_args())

original = cv2.imread(args["image"])

for gamma in np.arange(0.0, 8.5, 0.5):
    if gamma == 1: continue

    gamma = gamma if gamma > 0 else 0.1
    gamma_aplicado = aplicar_gamma(original, gamma=gamma)
    cv2.putText(gamma_aplicado, "gamma={}".format(gamma), (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)
    cv2.imshow("imgs", np.hstack([original, gamma_aplicado]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# python lab2_q1_gamma.py -i "imgs/jato.jpg"