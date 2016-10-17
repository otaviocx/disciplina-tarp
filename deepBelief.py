import numpy as np
import matplotlib.pyplot as plot
import math
from PIL import Image
from boltzmann import Layer


def binarize_color(pixel):
    total = np.sum(pixel)
    return 1 if total > 380 else 0

img = Image.open("digits.jpeg")
pix = img.load()

def get_digit(number):
    letter = []
    for j in range(0, 20):
        for i in range(0+number*20, 20+number*20):
            letter.append(binarize_color(pix[i, j]))
    return np.array(letter)


def to_image(letter):
    new_letter = []
    for k in letter:
        n = 0 if k <= 0 else 1
        new_letter.append([n, n, n])

    new_letter = np.array(new_letter)
    plot.imshow(new_letter.reshape((20, 20, 3)))

entradas = []
for i in range(0, 9):
    entradas.append(get_digit(i))

layerImg1 = Layer("Layer 1", 0.1, np.random.random_sample((400, 4)), entradas)
for i in range(0, 5000):
    print("============= Epoca", i+1, "=============")
    erro = layerImg1.run()
    print("Erro: ", erro)


weights = layerImg1.weights
weights = np.delete(weights, 0, 0)
weights = np.delete(weights, 0, 1)
weights = np.transpose(weights)

for i, w in enumerate(weights):
    plot.subplot(4, 3, i+1)
    to_image(w)
plot.show()

"""
layerImg2 = Layer("Layer 2", 0.1, np.random.random_sample((400, 10)), weights)
for i in range(0, 500):
    print("============= Epoca", i+1, "=============")
    erro = layerImg2.run()
    print("Erro: ", erro)

weights = layerImg2.weights
weights = np.delete(weights, 0, 0)
weights = np.delete(weights, 0, 1)
weights = np.transpose(weights)

for i, w in enumerate(weights):
    plot.subplot(3, 4, i+1)
    to_image(w)
plot.show()

layerImg3 = Layer("Layer 3", 0.1, np.random.random_sample((400, 10)), weights)
for i in range(0, 500):
    print("============= Epoca", i+1, "=============")
    erro = layerImg3.run()
    print("Erro: ", erro)

weights = layerImg3.weights
weights = np.delete(weights, 0, 0)
weights = np.delete(weights, 0, 1)
weights = np.transpose(weights)

for i, w in enumerate(weights):
    plot.subplot(3, 4, i+1)
    to_image(w)
plot.show()
"""