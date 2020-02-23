import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Aufgabe 1
def fizz_buzz(n):
    return ((n % 3 == 0) * 'Fizz' + (n % 5 == 0) * "Buzz") or n


# Aufgabe 2
def running_mean(l):
    return list(np.cumsum(l)/np.arange(1, len(l)+1))


# Aufgabe 3
def plot_data():
    data = pd.read_csv('Data_Task_3.csv', delimiter=';')
    l = [np.array(data[x].str.split(',').tolist()).astype(float) for x in ['A', 'B']]
    plt.plot(np.dot(*l)[0:6, :].T, '.', markersize=10)
    plt.show()


# Aufgabe 4
class Knoten():

    def __init__(self, key, kids):
        self.key = key
        self.kids = kids

    def add_kinder(self, kids):
        self.kids.extend(kids if isinstance(kids, list) else [kids])


if __name__ == '__main__':
    for i in range(1, 50):
        print(fizz_buzz(i))
    print("-----")
    print(running_mean([1, 2, 3, 4, 5]))
    print(running_mean([0, 4, -8, 4, 2]))
    print("-----")
    plot_data()
    print("-----")
    k1 = Knoten(1, [])
    k2 = Knoten(2, [])
    k3 = Knoten(3, [k1, k2])
    k4 = Knoten(4, [k1])
    k5 = Knoten(5, [k2])
    k1.add_kinder([k2, k4])
    k2.add_kinder(k4)
    for k in [k1, k2, k3, k4, k5]:
        print("Knoten %d - Kinder %s" % (k.key, [i.key for i in k.kids]))
