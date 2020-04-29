import sys
import numpy as np
import itertools

if __name__ == '__main__':
    # Use with "python items.py <file> <minfreq> <minconf>"
    file = sys.argv[1]
    minfreq, minconf = float(sys.argv[2]), float(sys.argv[3])

    with open(file) as ff:
        # Alle Warenkörbe
        data = np.array([line.strip('\n').split(" ") for line in ff.readlines()])

    # Alle Waren
    all_items = set([x for w in data for x in w])
    # Warensortierung
    item_order = {it: i for i, it in enumerate(all_items)}

    # alle Häufige Mengen mit ihrem Support
    h = dict()
    # häufige Waren
    f = [[i] for i in all_items]
    # Solange noch Kandidaten übrig sind
    while len(f) >= 0:
        # Support berechnen
        cnts = {tuple(i): len([1 for w in data if np.all(np.isin(i, w))]) for i in f}
        # Nur die häufige Mengen
        f = [i for i in f if cnts[tuple(i)] >= minfreq * len(data)]
        # Support merken
        h.update({tuple(i): cnts[tuple(i)] for i in f})
        # Mengen kombinieren (Sortierung beachten)
        f = [i1[:-1] + sorted([i1[-1], i2[-1]], key=item_order.get)
             for i1 in f for i2 in f if i1 != i2 and i1[:-1] == i2[:-1]]

    for Z, sup in h.items():
        # Alle Teilmengen von Z in aufsteigender Mächtigkeit
        A = [x for i in range(1, len(Z)) for x in list(itertools.combinations(Z, i))]
        while len(A) > 0:
            X = A.pop(-1)
            c = sup / h[X]
            if c >= minconf:
                print("[%s] -> [%s] %d %d" % (",".join(set(X)), ",".join(set(Z) - set(X)), sup, c))
            else:
                # Teilmengen von X werden aussortiert
                A = [i for i in A if not np.all(np.isin(i, X))]
