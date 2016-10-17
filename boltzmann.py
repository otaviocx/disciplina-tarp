import numpy as np
import matplotlib.pyplot as plot
import math


def sigmoid(u):
    return 1 / (1 + math.exp(-u))
sigmoid_matrix = np.vectorize(sigmoid)


def digitize(x, y):
    return 1 if (x >= y) else 0
digitize_matrix = np.vectorize(digitize)


class Layer(object):

    def __init__(self, name, learning_rate, weights, inputs):
        self.name = name
        self.weights = np.append([np.zeros(len(weights[0]))], weights, 0)
        self.weights = np.append(np.zeros((len(self.weights), 1)), self.weights, 1)
        self.inputs = np.append(np.ones((len(inputs), 1)), inputs, 1)
        self.learning_rate = learning_rate
        self.activation_energy = []
        self.neuron_association = []
        self.positive_association = []
        self.negative_association = []
        self.gUh_minus = []

    def set_inputs(self, inputs):
        self.inputs = inputs

    # +Passo 1: Calcular a energia de ativação para cada i-esimo neurônio oculto
    def calculate_activation_energy(self):
        self.activation_energy = np.dot(self.inputs, self.weights)
        return self.activation_energy

    # +Passo 2: Calcular a probabilidade de cada entrada ser associada a um neurônio oculto
    def calculate_neuron_association(self):
        self.calculate_activation_energy()
        self.neuron_association = sigmoid_matrix(self.activation_energy)
        self.positive_association = np.dot(np.transpose(self.inputs), self.neuron_association)
        #print("Associações Positivas")
        #print(self.positive_association)
        self.neuron_association = self.random_round(self.neuron_association)

    # +Passo 2.1: Calcular uma matrix com aleatoriedade baseada na probabilidade
    def random_round(self, association):
        random_matrix = np.random.random_sample((len(association), len(association[0])))
        return digitize_matrix(association, random_matrix)

    # Método auxiliar que calcula um passo na fase negativa
    @staticmethod
    def revert_outputs(association, weights):
        u_matrix = np.dot(association, weights)
        return sigmoid_matrix(u_matrix)

    # Calcular as associações negativas
    def negative_phase(self):
        # Realiza a fase positiva
        self.calculate_neuron_association()

        # Passos 1 e 2 da fase negativa
        self.GUh_minus = self.revert_outputs(self.neuron_association, np.transpose(self.weights))
        # Corrigindo o bias para 1.0
        for row in self.GUh_minus:
            row[0] = 1

        # Passos 3 e 4 da fase negativa
        GUv_minus = self.revert_outputs(self.GUh_minus, self.weights)

        # Passo 5: Calculo das associações negativas
        self.negative_association = np.dot(np.transpose(self.GUh_minus), GUv_minus)
        #print("Associações Negativas")
        #print(self.negative_association)

    # Executa as fases positiva e negativa e reajusta os pesos
    def recalculate_weights(self):
        self.negative_phase()
        self.weights += self.learning_rate * (self.positive_association - self.negative_association) / len(self.inputs)
        print("Novos Pesos")
        print(self.weights)

    # Executa as fases positiva e negativa, reajusta os pesos e calcula e retorna o erro.
    def run(self):
        self.recalculate_weights()
        summation = 0
        for i, row in enumerate(self.GUh_minus):
            sample = self.inputs[i]
            error_array = sample - row
            summation += np.sum(error_array * error_array)
        return summation

layer1 = Layer("L1", 0.1, [
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6],
    [0.7, 0.8],
    [0.9, 1],
    [0.1, 0.2]
], [
    [0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 1],
    [0, 1, 1, 1, 0, 0],
    [1, 0, 0, 0, 1, 1]
])
for i in range(0, 5000):
    print("============= Epoca", i+1, "=============")
    erro = layer1.run()
    print("Erro: ", erro)
