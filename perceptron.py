def degrau(u):
	if(u < 0): 
		return 0
	else:
		return 1
		
def perceptron(entradas, pesos, bias, limiar):
	if(len(entradas) != len(pesos)):
		return False
	
	potencial = bias - limiar
	for i, e in enumerate(entradas):
		potencial += e * pesos[i]
		
	return degrau(potencial)
	
def calcularPesos(entradas, pesos, erro, taxa):
	if(len(entradas) != len(pesos)):
		return False
	
	novosPesos = []
	for i, e in enumerate(entradas):
		delta = taxa*erro*e;
		novosPesos.append(pesos[i]+delta)
		
	return novosPesos

def treinar(entradas, desejados, erroAceitavel, maxTentativas, pesosIniciais, bias):
	pesos = pesosIniciais
	print("Pesos iniciais:", pesos)
	
	erroTotal = 1
	tentativa = 1
	while(erroTotal > erroAceitavel and tentativa <= maxTentativas):
		erroTotal = 0;
		print("\nTentativa ", tentativa, ":")
		for i, entrada in enumerate(entradas):
			obtido = perceptron(entrada, pesos, bias, 0)
			print("Entradas:", entrada, "| Obtido:", obtido, "| Desejado:", desejados[i])
			erro = abs(desejados[i] - obtido)
			erroTotal += erro
			if(erro > erroAceitavel):
				pesos = calcularPesos(entrada, pesos, erro, 1)
				print("Mudando pesos para:", pesos)
		tentativa += 1
	print("\nPesos finais:", pesos)
	

entradas = [
	[0,0],
	[0,1],
	[1,0],
	[1,1]
]
desejados = [0,1,1,1]
treinar(entradas, desejados, 0.00001, 10, [-5, -5], -0.8649)
