import numpy as np
import matplotlib.pyplot as plot


def degrau(u):
    return 0 if u < 0 else 1


def limiar2D(pesos, bias, axis):
    x = np.arange(-1, 3, 1)
    y = [-(pesos[0]/pesos[1])*i-(bias/pesos[1]) for i in x]
    return y if axis == 'y' else x


def perceptron(entradas, pesos, bias, limiar):
    if len(entradas) != len(pesos):
        return False

    potencial = bias - limiar + np.sum(entradas*pesos)
    return degrau(potencial)


def calcularPesos(entradas, pesos, erro, taxa):
    if len(entradas) != len(pesos):
        return False
    
    return pesos+taxa*erro*entradas


def treinar(entradas, desejados, erroAceitavel, maxTentativas, pesosIniciais, bias, taxa):
    pesos = pesosIniciais
    print("Pesos iniciais:", pesos)

    plot.figure(1)

    erroTotal = 1
    tentativa = 1
    while erroTotal > erroAceitavel and tentativa <= maxTentativas:
        erroTotal = 0
        print("\nTentativa ", tentativa, ":")
        plot.subplot(220+tentativa)
        plot.axis([-1, 2, -1, 2])
        plot.plot(limiar2D(pesos, bias, 'x'), limiar2D(pesos, bias, 'y'))
        for i, entrada in enumerate(entradas):
            plot.plot(entrada[0], entrada[1], marker='o')
            obtido = perceptron(entrada, pesos, bias, 0)
            print("Entradas:", entrada, "| Obtido:", obtido, "| Desejado:", desejados[i])
            erro = desejados[i] - obtido
            erroTotal += abs(erro)
            if(abs(erro) > erroAceitavel):
                pesos = calcularPesos(entrada, pesos, erro, taxa)
                print("Mudando pesos para:", pesos)
        tentativa += 1
    print("\nPesos finais:", pesos)
    plot.show()

entradas = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
desejados = np.array([0, 1, 1, 1])
treinar(entradas, desejados, 0.00001, 10, [0.3192, 0.3129], -0.8649, 1)
