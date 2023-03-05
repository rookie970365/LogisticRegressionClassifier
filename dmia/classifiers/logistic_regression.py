import numpy as np
from scipy import sparse


class LogisticRegression:
    def __init__(self):
        self.w = None
        self.loss_history = None

    def train(self, X, y, learning_rate=0.001, reg=0.00001, num_iters=100, batch_size=200, verbose=False):
        """
        Метод для обучения классификатора, использующий стохастический градиентный спуск.

        Inputs:
        - X: N x D массив обучающих данных. Каждая точка обучения представляет собой D-мерный столбец.
        - y: одномерный массив длины N с метками 0-1 для 2 классов.
        - learning_rate: float  скорость обучения для оптимизации.
        - reg: float  сила регуляризации.
        - num_iters: integer  количество итераций при оптимизации.
        - batch_size: integer количество обучающих примеров для использования на каждом шаге.
        - verbose: boolean  выводить ход выполнения оптимизации, если значение true.

        Outputs:
        Список, содержащий значение функции потерь на каждой итерации обучения.
        """

        # Add a column of ones to X for the bias sake.
        X = LogisticRegression.append_biases(X)
        num_train, dim = X.shape
        if self.w is None:
            # lazily initialize weights
            self.w = np.random.randn(dim) * 0.01

        # Run stochastic gradient descent to optimize W
        self.loss_history = []
        for it in range(num_iters):

            #########################################################################
            # Выборка случайных элементов в количестве batch_size из обучающих данных
            # и соответствующих им меток для использования в этом раунде градиентного
            # спуска.

            indices = np.random.choice(num_train, batch_size)
            X_batch = X[indices, :]
            y_batch = y[indices]
            #########################################################################

            # evaluate loss and gradient
            loss, gradW = self.loss(X_batch, y_batch, reg)
            self.loss_history.append(loss)

            #########################################################################
            # Обновление весов, используя градиент и скорость обучения

            self.w -= learning_rate * gradW
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return self

    def predict_proba(self, X, append_bias=False):
        """
        Метод для прогнозирования вероятностей для точек данных, использующий обученные
        веса линейного классификатора

        Inputs:
        - X: N x D   массив данных. Каждая строка представляет собой D-мерную точку.
        - append_bias: boolean  добавление смещения перед прогнозированием.

        Outputs:
        - y_proba:  двумерный массив (N, 2) вероятностей классов для данных в X, каждая
        строка представляет собой распределение классов [prob_class_0, prob_class_1].
        """

        if append_bias:
            X = LogisticRegression.append_biases(X)

        ###########################################################################
        predictions = self.sigmoid(X.dot(self.w.T))
        y_proba = np.vstack([1 - predictions, predictions]).T
        ###########################################################################
        return y_proba

    def predict(self, X):
        """
        Метод для прогнозирования меток для точек данных, использующий метод ``predict_proba``
        Inputs:
        - X: N x D массив обучающих данных. Каждый столбец представляет собой D-мерную точку.

        Outputs:
        - y_pred: предсказанные метки для данных в X. y_pred - это одномерный массив длиной N,
         и каждый элемент является целым числом, дающим предсказанный класс.
        """

        ###########################################################################
        y_proba = self.predict_proba(X, append_bias=True)
        y_pred = y_proba.argmax(axis=1)
        ###########################################################################
        return y_pred

    @staticmethod
    def sigmoid(z):
        # Функция сигмоиды
        return 1.0 / (1.0 + np.exp(-z))

    def loss(self, X_batch, y_batch, reg):
        """
        Функция потерь логистической регрессии

        Inputs:
        - X: N x D массив данных. Данные представляют собой D-мерные строки
        - y: одномерный массив длины N с метками 0-1, для 2 классов

        Outputs:
        a tuple of:
        - loss: в виде одиночного float
        - dw:  градиент по отношению к весам w, массив такой же формы как w
        """

        dw = np.zeros_like(self.w)  # initialize the gradient as zero
        loss = 0
        # Compute loss and gradient. Your code should not contain python loops.
        a = self.sigmoid(X_batch.dot(self.w))
        dw = (a - y_batch) * X_batch
        loss = -np.dot(y_batch, np.log(a)) - (1.0 - y_batch).dot(np.log(1.0 - a))
        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead so we divide by num_train.
        # Note that the same thing must be done with gradient.
        num_train = X_batch.shape[0]
        dw /= num_train
        loss /= num_train
        # Add regularization to the loss and gradient.
        # Note that you have to exclude bias term in regularization.
        dw[:-1] += (self.w[:-1] * reg) / num_train
        loss += (reg / (num_train * 2.0)) * self.w[:-1].dot(self.w[:-1])
        return loss, dw

    @staticmethod
    def append_biases(X):
        return sparse.hstack((X, np.ones(X.shape[0])[:, np.newaxis])).tocsr()
