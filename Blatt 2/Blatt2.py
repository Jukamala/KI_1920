import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load data from csv
def data_loader():
    return [np.transpose(pd.read_csv('%s_Dataset.csv' % f, sep=' ', header=None).to_numpy()) for f in ['Train', 'Test']]


# Plot all train and test data
def plot_data(x_train, y_train, x_test, y_test):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].scatter(x_train, y_train, s=4)
    ax[0].set_title('Trainingsdaten')
    ax[1].scatter(x_test, y_test, s=4)
    ax[1].set_title('Testdaten')
    fig.suptitle('Data from %d Samples' % len(x_train))
    plt.show()


# plot the predicted data
def plot_predict(model, x, y):
    prediction = model.predict(x)
    fig, ax = plt.subplots(figsize=(10, 5))
    dist = -np.abs(y - prediction)
    norm = plt.Normalize(np.min(dist), np.max(dist))
    ax.scatter(x, y, s=4, c=plt.cm.viridis(norm(dist)))
    ax.plot(x, prediction, c='red')
    ax.set_title(r'Model: $%s$' % ''.join(['+ %.2f\cdot x^{%d}' % (a, i) if i > 0 and a >= 0 else
                                           '%.2f\cdot x^{%d}' % (a, i) if i > 0 else
                                           '%.2f' % a for i, a in enumerate(model.model)]))

    plt.show()


class RegressionModel:
    def __init__(self, degree):
        self.degree = degree
        self.model = None

    def fit(self, x_train, y_train):
        # Normalengleichung
        X = self._generate_features(x_train)
        self.model = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y_train))

    def predict(self, x_test):
        if self.model is None:
            raise RuntimeError('fit a model using .fit(...) before calling .predict(...)')

        # Horner-Schema für kleinere Fehler
        res = 0
        for a in np.flip(self.model):
            res = x_test * res + a
        return res

    def _generate_features(self, x_values):
        p, x_val = np.meshgrid(np.arange(0, self.degree + 1), x_values)
        return x_val**p




if __name__ == '__main__':
    [x_train, y_train], [x_test, y_test] = data_loader()
    #plot_data(x_train, y_train, x_test, y_test)

    for p in range(2, 5):
        model = RegressionModel(p)
        model.fit(x_train, y_train)
        print(model.model)
        plot_predict(model, x_train, y_train)


"""
#### example usage 
...
p2 = RegressionModel(2)
p2.fit(x_train, y_train)
y_predict = p2.predict(x_test)
...
"""