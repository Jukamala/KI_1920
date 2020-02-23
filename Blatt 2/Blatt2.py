import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Load data from csv
def data_loader():
    return [pd.read_csv('%s_Dataset.csv' % f, sep=' ', header=None).to_numpy().T for f in ['Train', 'Test']]


# Plot all train and test data
def plot_data(x_train, y_train, x_test, y_test):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].scatter(x_train, y_train, s=4)
    ax[0].set_title('Trainingsdaten')
    ax[1].scatter(x_test, y_test, s=4)
    ax[1].set_title('Testdaten')
    fig.suptitle('Data from %d Samples' % len(x_train))
    plt.show()


# plot data + predicted and real labels
# returns predictions
def plot_predict(model, x, y):
    fig, ax = plt.subplots(figsize=(10, 5))
    y_predict = model.predict(x)
    dist = np.abs(y - y_predict)
    norm = plt.Normalize(np.min(dist), np.max(dist))
    ax.scatter(x, y, s=4, c=plt.cm.viridis_r(norm(dist)))
    ax.plot(x, y_predict, c='red', label='y_predict [ED = %.2f]' % ed(y, y_predict))
    ax.legend(loc='lower left')
    ax.set_title(r'Model: $%s$' % ''.join(['+ %.2f x^{%d}' % (a, i) if i > 0 and a >= 0 else
                  '%.2f x^{%d}' % (a, i) for i, a in enumerate(model.model)]).replace('x^{0}','').replace('x^{1}','x'))
    cax = make_axes_locatable(ax).append_axes('right', size='2%', pad=0.05)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis_r'), cax=cax).set_label('distance to prediction')
    plt.show()
    return y_predict


# Combine all predictions in a nice plot
def overview(x_train, y_train, x_test, y_test, predictions):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x_train, y_train, s=4, c='darkcyan', label='train_data')
    ax.scatter(x_test, y_test, s=4, c='turquoise', label='test_data')
    for c, [i, p] in zip(['red', 'orangered', 'darkorange'], enumerate(predictions)):
        ax.plot(x_test, p, c=c, label='Prediction with degree %d [ED = %.2f]' % (i+2, ed(y_test, p)))
    ax.legend(loc='lower left')
    plt.show()


# Euclidian Distance
def ed(y, y_hat):
    return np.sqrt(np.sum((y - y_hat)**2))


# Mean Square Error
def mse(y, y_hat):
    return np.mean((y - y_hat)**2)


class RegressionModel:
    def __init__(self, degree):
        self.degree = degree
        self.model = None
        self.Normalizers = [plt.Normalize(), plt.Normalize()]

    def fit(self, x_train, y_train, mu=1, tol=0.00001):
        # Normalize data
        self.Normalizers[0].autoscale(x_train)
        self.Normalizers[1].autoscale(y_train)

        # Normalize
        x_train = np.array(self.Normalizers[0](x_train))
        y_train = np.array(self.Normalizers[1](y_train))

        # Features
        X = self._generate_features(x_train)

        # Normalengleichung as a starting point
        try:
            self.model = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y_train))
        except np.linalg.LinAlgError:
            # Start with Gaussian distribiution
            self.model = np.random.normal(0, 1, self.degree + 1)

        # Fine-Tune this with the Gradient Descent to combat rounding errors
        gradient = np.matmul(X.T, np.matmul(X, self.model)) - np.matmul(y_train.T, X)
        steps = 0
        while np.linalg.norm(gradient) > tol:
            gradient = np.matmul(X.T, np.matmul(X, self.model)) - np.matmul(y_train.T, X)

            # Get a status every 1.000 Steps
            steps += 1
            if steps == 1000:
                steps = 0
                print("Norm: %.4f\nMSE: %.4f" % (np.linalg.norm(gradient), mse(self.predict(x_train), y_train)))

            self.model = self.model - mu * (1/len(x_train)) * gradient

    def predict(self, x_test):
        # Normalized on x_train
        x_test = np.array(self.Normalizers[0](x_test))

        if self.model is None:
            raise RuntimeError('fit a model using .fit(...) before calling .predict(...)')

        # Horner-Schema für kleinere Fehler
        res = 0
        for a in np.flip(self.model):
            res = x_test * res + a

        # De-normalize
        return np.array(self.Normalizers[1].inverse(res))

    def _generate_features(self, x_values):
        p, x_val = np.meshgrid(np.arange(0, self.degree + 1), x_values)
        return x_val**p


if __name__ == '__main__':
    [x_train, y_train], [x_test, y_test] = data_loader()
    plot_data(x_train, y_train, x_test, y_test)

    predictions = []
    for p in range(2, 5):
        model = RegressionModel(p)
        model.fit(x_train, y_train)
        plot_predict(model, x_train, y_train)
        predictions += [plot_predict(model, x_test, y_test)]
    overview(x_train, y_train ,x_test, y_test, predictions)

    y = np.array([0.8, 0.43, 1.74, 0.26, 4.06, 0.73, 2.8, 3.37])
    y_hat = np.array([3.49, 1.3, 1.49, 4.12, 2.19, 4.24, 4.67, 0.22])
    print('ED(y,y_hat) = %.2f' % ed(y, y_hat))