import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#carregando o dataset
dados = pd.read_csv('diabetes.csv')


X = dados.drop('Outcome', axis=1).values
y = dados['Outcome'].values

#numero de epocsa
numEpocas = 1000
q = len(y)

#parametros
eta = 0.05
m = 8
N = 16
L = 1

#inicializando matrizes de peso
W1 = np.random.random((N, m + 1))
WU = np.random.random((N, N + 1))
WX = np.random.random((N, N + 1))
W2 = np.random.random((L, N + 1))

#arrays de erro
E = np.zeros(q)
Etm = np.zeros(numEpocas)

#bias
bias = 1

#ativação Sigmóide
def sigmoid(X):
    return 1 / (1 + np.exp(-X))

#treinamento
for i in range(numEpocas):
    for j in range(q):
        Xb = np.hstack((bias, X[j]))

        o1 = sigmoid(W1.dot(Xb))
        o1b = np.insert(o1, 0, bias)
        o2 = sigmoid(WU.dot(o1b))
        o2b = np.insert(o2, 0, bias)
        o3 = sigmoid(WU.dot(o2b))
        o3b = np.insert(o3, 0, bias)

        Y = sigmoid(W2.dot(o1b))

        e = y[j] - Y
        E[j] = (e.transpose().dot(e)) / 2

        delta2 = np.diag(e).dot((1 - Y * Y))
        vdelta2 = (W2.transpose()).dot(delta2)
        deltaU = np.diag(1 - o3b * o3b).dot(vdelta2)
        vdeltaU = (WU.transpose()).dot(deltaU[1:])
        deltaX = np.diag(1 - o2b * o2b).dot(vdeltaU)
        vdeltaX = (WX.transpose()).dot(deltaX[1:])
        delta1 = np.diag(1 - o1b * o1b).dot(vdeltaX)

        W1 = W1 + eta * (np.outer(delta1[1:], Xb))
        WU = WU + eta * (np.outer(deltaU[1:], o3b))
        WX = WX + eta * (np.outer(deltaX[1:], o2b))
        W2 = W2 + eta * (np.outer(delta2, o1b))

    Etm[i] = E.mean()

#plotar o erro de treinamento
plt.xlabel("Épocas")
plt.ylabel("Erro Médio")
plt.plot(Etm, color='b')
plt.plot(Etm)
plt.show()

#teste
Erro_Teste = np.zeros(q)

for i in range(q):
    Xb = np.hstack((bias, X[i]))

    o1 = sigmoid(W1.dot(Xb))
    o1b = np.insert(o1, 0, bias)
    o2 = sigmoid(WU.dot(o1b))
    o2b = np.insert(o2, 0, bias)
    o3 = sigmoid(WU.dot(o2b))
    o3b = np.insert(o3, 0, bias)

    Y = sigmoid(W2.dot(o1b))
    print(Y)

    Erro_Teste[i] = y[i] - (Y)

print(Erro_Teste)
print(np.round(Erro_Teste) - y)

erros = 0

for i in range(len(Erro_Teste)):
    if np.round(Erro_Teste[i]) != 1:
        erros += 1

print(np.round(Erro_Teste))
print('Erros:', erros)
print("Erros: " + str(np.round(Erro_Teste)))
print("Porcentagem: {:.2f}%".format((erros * 100) / q))
