from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


# Aufgabe 4.1
def main():
    # load data
    x_test, x_train, y_test, y_train = map(lambda x: np.load(x + ".npy"), ["x_test", "x_train", "y_test", "y_train"])

    # preprocessing
    x_test, x_train = map(lambda x: np.reshape(x, (x.shape[0], -1)), [x_test, x_train])
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_test, x_train = map(lambda x: scaler.transform(x), [x_test, x_train])

    # Train a neural net (takes < 10 iterations)
    net = MLPClassifier(solver="adam", alpha=1e-2, hidden_layer_sizes=(180, 80, 50), random_state=1,
                        learning_rate_init=0.0005, learning_rate="adaptive", batch_size=50, max_iter=100, tol=1e-2,
                        early_stopping=True, n_iter_no_change=3, verbose=True, activation="relu")
    net.fit(x_train, y_train)
    print(classification_report(net.predict(x_test), y_test))
    print(net.score(x_test, y_test))
    disp = plot_confusion_matrix(net, x_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    plt.show()


"""
Aufgabe 4.2)
a) Daten in Klasse A: 100 + 2 = 102 
   Daten in Klasse B: 8 + 5   = 13  
   
b) Klasse A: Precision: 100 / (100 + 8)              = 92.59 %
             Recall :   100 / (100 + 2)              = 98.04 %
             F1:        2 / (1/Precision + 1/Recall) = 95.24 %
   Klasse B: Precision: 5 / (5 + 2)                  = 71.43 %
             Recall :   5 / (5 + 8)                  = 38.46 %
             F1:        2 / (1/Precision + 1/Recall) = 50.00 %
   Accuracy:            (100 + 5) / (102 + 13)       = 91.30 %
             
c) Weder Precision noch Recall alleine kann die Qualität des Klassifiers gut beschreiben.
   Beim Recall werden False Positves und bei der Precision False Negatives vernachlässigt.
   Wenn man z.B. jeden Datenpunkt der Klasse A zuordnet erhält man einen Recall von 100%
   und wenn man nur den einen Wert der Klasse A zuordnet von dem man am sichersten ist,
   das er in A liegt erhält man eine Precision von 100%. Beide Classifier wären sehr schlecht.
   Der F1-Score berücksichtigt beide und ist so ein gutes Mittel um die Qualität zu schätzen.
   Accuracy ist keine qute Metrik, da sie bei nicht ähnlich großen Klassen (wie hier 102 vs 13)
   die große Klasse viel höher gewichtet.
   
d) Vernachlässigt man die Varianz der Gleichverteilung erhält man in Erwartung die Confusionmatrix von K2:

     |  A  |  B
   A |  51 | 6.5
   B |  51 | 6.5
   
   Klasse A: Precision: 51 / (51 + 6.5)              = 88.70 %
             Recall :   51 / (51 + 51)               = 50.00 %
             F1:        2 / (1/Precision + 1/Recall) = 63.95 %
   Klasse B: Precision: 6.5 / (6.5 + 51)             = 11.30 %
             Recall :   6.5 / (6.5 + 6.5)            = 50.00 %
             F1:        2 / (1/Precision + 1/Recall) = 18.44 %
   Accuracy:            (51 + 6.5) / (102 + 13)      = 50.00 %
   
e) Vergleicht man die Precision, Recall und F1-Score von K1 und K2 für die Klasse B
   fällt auf, das K2 zwar einen besseren Recall hat, die Precision dafür aber viel schlechter ist.
   Dies spiegelt sich auch im F1-Score wieder, es folgt, dass K1 besser geignet ist,
   wenn man die gegebene Verteilung von As und Bs gegeben hat (also deutlich mehr A's als B's)
   
f) K1 lässt sich verbessern, in dem der schlechte Recall von B verbessert wird,
   also mehr B's auch als B's erkannt werden.
"""


if __name__ == '__main__':
    main()