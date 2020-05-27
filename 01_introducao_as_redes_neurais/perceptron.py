#%% [markdown]
# Importando pacotes
  
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

#%% [markdown]
# Definição de função para visualização de exemplo de teste do Perceptron
  
def plot_data(feature, colors, weight=None):

    plt.scatter(feature[:, 0], feature[:, 1], c=colors)

    if weight is not None:
        px = [0, 0]
        py = [0, 0]
        py[0] = -weight[0] / weight[2]
        px[1] = -weight[0] / weight[1]
        plt.plot(px, py, linestyle='solid')

#%% [markdown]
# Definição da classe Perceptron, baseada na classe BaseEstimator
class Perceptron(BaseEstimator, TransformerMixin):
    def __init__(self, lr=1, epoch=500):
        self.lr = lr
        self.epoch = epoch
        pass

    def fit(self, X, y=None):
        X = np.array(X)
        bias = np.ones((X.shape[0], 1))
        X = np.hstack((bias, X))
        self.weights = np.random.rand(X.shape[1])

        for ep in range(self.epoch):
            for idx, _ in enumerate(X):

                xi = X[idx]
                yi = y[idx]
                pred = np.sign(np.dot(xi, self.weights))
                dist = yi - pred
                self.weights = self.weights + self.lr * dist * xi

        return self

    def predict(self, X, y=None):
        X = np.array(X)
        shapelen = len(X.shape)
        X = X if shapelen > 1 else np.array([X])
        bias = np.ones((X.shape[0], 1))
        X = np.hstack((bias, X))

        pred = np.sign(np.dot(X, self.weights))
        return pred if shapelen > 1 else pred[0]

#%% [markdown]
#  Teste da perceptron com dataset resultante da operação binária OR
  
data = np.array([
    [0, 0, -1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

X_train = data[:, 0:2]
y_train = data[:, 2]

perceptron = Perceptron()
perceptron.fit(X_train, y_train)
plot_data(X_train, y_train, perceptron.weights)

#%% [markdown]
# Teste do perceptron com o dataset load_breast_cancer  

data = load_breast_cancer()
X, y = data.data, data.target

#Separando o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#%% [markdown]
# Transformando as classes binarias do dataset nas classes 1 e -1
  
labelbinarizer = LabelBinarizer(neg_label=-1)
labelbinarizer.fit(y_train)
y_train_bin = labelbinarizer.transform(y_train).reshape(-1)
y_test_bin = labelbinarizer.transform(y_test).reshape(-1)

#%% [markdown]
# Treinando o perceptron
  
perceptron = Perceptron()
perceptron.fit(X_train, y_train_bin)

#%% [markdown]
# Acurácia da predição dos dados de treinamento
  
y_train_pred = perceptron.predict(X_train)
accuracy_score(y_train_bin, y_train_pred)

#%% [markdown]
# Acurácia da predição dos dados de teste
  
y_test_pred = perceptron.predict(X_test)
accuracy_score(y_test_bin, y_test_pred)