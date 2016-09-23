import numpy as np
import matplotlib.pyplot as plot
import math


def degrau(u):
    return 0 if u < 0 else 1


def sigmoide(u):
    return 1 / (1+math.exp(-u))


def limiar2D(pesos, bias, axis):
    x = np.arange(-1, 3, 1)
    y = [-(pesos[0]/pesos[1])*i-(bias/pesos[1]) for i in x]
    return y if axis == 'y' else x


def perceptron(entradas, pesos, bias, limiar):
    if len(entradas) != len(pesos):
        return False

    potencial = bias - limiar + np.sum(entradas*pesos)
    return sigmoide(potencial), potencial


def calcularPesos(entradas, pesos, bias, erro, taxa):
    if len(entradas) != len(pesos):
        return False

    return bias + taxa * erro, pesos + taxa * erro * entradas


def plotarEntradas(entradas, plot):
    for entrada in entradas:
        plot.plot(entrada[0], entrada[1], marker='o')


def plotarGrafico(plot, tentativa, entradas, pesos, bias):
    plot.subplot(3, 4, math.ceil(math.log(tentativa, 2))+1)
    plot.subplots_adjust(hspace=0.5)
    plot.grid()
    plot.title("Tentativa " + repr(tentativa))
    plot.axis([-1, 2, -1, 2])
    plot.plot(limiar2D(pesos, bias, 'x'), limiar2D(pesos, bias, 'y'))
    plotarEntradas(entradas, plot)


def treinar(entradas, desejados, erroAceitavel, maxTentativas, pesosIniciais, biasInicial, taxa):
    pesos = pesosIniciais
    bias = biasInicial
    print("Pesos iniciais:", pesos)

    plot.figure(1)

    erroAnterior = 2
    erroTotal = 1
    tentativa = 1
    plotInterval = 1
    while abs(erroTotal) > erroAceitavel and tentativa <= maxTentativas and erroAnterior-erroTotal > 0.000000001:
        erroAnterior = erroTotal
        erroTotal = 0
        print("\nTentativa ", tentativa, ":")

        if tentativa % plotInterval == 0:
            plotarGrafico(plot, tentativa, entradas, pesos, bias)
            plotInterval *= 2

        for i, entrada in enumerate(entradas):
            obtido, u = perceptron(entrada, pesos, bias, 0)
            print("Entradas:", entrada, "| Obtido:", obtido, "| Desejado:", desejados[i])
            erro = (desejados[i] - obtido)
            erroTotal += erro * erro
            if(abs(erro) > erroAceitavel):
                print("Erro atual:", erro)
                bias, pesos = calcularPesos(entrada, pesos, bias, erro, taxa)
                print("Mudando pesos para:", pesos, "| Bias:", bias)

        erroTotal /= len(entradas)
        print("Erro total:", erroTotal, "| Variação:", erroAnterior-erroTotal)
        tentativa += 1
    print("\nPesos finais:", pesos, "| Bias final:", bias)
    plotarGrafico(plot, tentativa, entradas, pesos, bias)
    plot.show()

entradas = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
desejados = np.array([0, 1, 1, 1])
treinar(entradas, desejados, 0.00001, 1000, np.random.randn(2), 2, 1.0)
# [0.3192, 0.3129]
# -0.8649