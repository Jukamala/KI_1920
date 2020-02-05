import numpy as np
import time


class my_Bot:
    def __init__(self, spieler_farbe):
        self.spieler_farbe = spieler_farbe
        if self.spieler_farbe == "black":
            self.gegner_farbe = "white"
        else:
            self.gegner_farbe = "black"
        self.spielfeld = None
        self.pos_felder = None
        self.cur_choice = None
        self.timeout = False

        # Store evaluated positions - (value, value_type)
        self.positions = dict()

    def set_next_stone(self):
        # empty - 0, white - 1, black - 2
        sp = np.zeros((8, 8))
        sp[self.spielfeld == 'white'] = 1
        sp[self.spielfeld == 'black'] = 2

        pos_zeilen = [(idr, idz) for idr, row in enumerate(self.pos_felder) for idz, zelle in enumerate(row) if
                      zelle == "pos_f"]
        self.cur_choice = pos_zeilen[
            0]  # Bei jedem Bot zuerst irgendein Wert setzen, falls Timeout abgelaufen, wird dieser Wert genommen
        print(self.spielfeld)
        print(self.spieler_farbe)
        print(self.pos_felder)


    def alpha_beta(self, position, spieler, depth, alpha=float('-inf'), beta=float('inf')):
        """
        position:  8x8 array with 0 - empty, 1 - white, 2 - black
        spieler:   True - White, False - Black
        depth:     # of pieces on board (depth of search)

        returns next move by calculating best end-value
        value is white number of pieces - black number of pieces
        -> white is max, black is min
        """
        if position in self.positions:
            return self.positions[position][0]

        if depth == 64:
            return

        moves = self.possible_felder(white)
        int i,n,value,localalpha=alpha,bestvalue=-INFINITY;
        int hashvalue;

        if(lookup(p, depth, &alpha, &beta, &hashvalue))
            return hashvalue;

        if(checkwin(p))
            return -INFINITY;

        if(depth == 0)
            return evaluation(p);

        n = makemovelist(p,list);
        if(n==0)
            return handlenomove(p);

        for(i=0;i<n;i++)
            {
            domove(list[i],&p);
            value = -alphabeta(p,depth-1,-beta,-localalpha);
            undomove(list[i],&p);
            bestvalue = max(value,bestvalue);
            if(bestvalue >= beta)
                break;
            if(bestvalue>localalpha)
                localalpha = bestvalue;
            }

        store(p, depth, bestvalue, alpha, beta);

        return bestvalue;
        }

    def minmax(g, node, spieler, alpha=float('-inf'), beta=float('inf')):

    if
    if max_player:
        value = float('-inf')
        for n in g[node]:
            value = max(value, minmax(g, n, False, alpha=alpha, beta=beta))
            alpha = max(alpha, value)
            if alpha > beta:
                break
    else:
        value = float('inf')
        for n in g[node]:
            value = min(value, minmax(g, n, True, alpha=alpha, beta=beta))
            beta = min(beta, value)
            if alpha >= beta:
                break
    # Also set alpha/beta in g
    nx.set_node_attributes(g, {node: {'alpha': alpha, 'beta': beta}})
    return value

    def check_rules(self, neuer_stein, spieler):
        """
        :param neuer_stein: Tupel von der Form (x,y) mit x und y im Intervall [0,7]. Wobei x der Spalte
                            und y der Zeile entspricht. Beispiel: self.spielfeld[y][x]
                            Zum aktualisieren des Spielfeldes genügt es, wenn sie das Array self.spielfeld aktualisieren, das
                            heißt, Sie müssen nicht die Methode update_spielfeld aufrufen.
        :param spieler: Entspricht dem aktuellen Spieler 0 steht für "black" 1 für "white"
        :param check_only: optinaler Parameter.
                            Wenn der Zug zulässig ist und check_only=False, dann soll self.spielfeld
                            aktualisiert werden. check_only=True, dann wird self.spielfeld nicht aktualisiert
        :return: True, wenn neuer_stein für spieler ein zulässiger Zug ist, Flase sonst
        """
        idx, idy = neuer_stein
        akt_farbe = "black" if spieler == 0 else "white"
        gegner_farbe = "black" if ((spieler + 1) % 2) == 0 else "white"

        # print("\nidy, idx =", idy, idx)
        # print("self.spielfeld[idy][idx] =", self.spielfeld[idy][idx])

        if not self.spielfeld[idy][idx] == "empty":
            return False
        else:
            possible_directions = []  # von (-1,-1) bis (1,1)
            for row in np.arange(max(idy - 1, 0), min(idy + 2, 8), 1):
                for zelle in np.arange(max(idx - 1, 0), min(idx + 2, 8), 1):
                    if not (row == idy and zelle == idx) and self.spielfeld[row][zelle] == gegner_farbe:
                        # print("(row-idy, zelle-idx) =", (row-idy, zelle-idx))
                        possible_directions.append((row - idy, zelle - idx))
            if len(possible_directions) > 0:
                richtiger_zug = False
                for elem in possible_directions:
                    schluss_stein_gefunden = False
                    temp_spielfeld = self.spielfeld.copy()
                    y_direction = idy + elem[0]
                    x_direction = idx + elem[1]
                    while y_direction in range(8) and x_direction in range(8) and not schluss_stein_gefunden:
                        if self.spielfeld[y_direction][x_direction] == gegner_farbe:
                            temp_spielfeld[y_direction][x_direction] = akt_farbe
                        elif self.spielfeld[y_direction][x_direction] == akt_farbe:
                            schluss_stein_gefunden = True
                            richtiger_zug = True
                            if not check_only:
                                self.spielfeld = temp_spielfeld.copy()
                        elif self.spielfeld[y_direction][x_direction] == "empty":
                            break

                        y_direction += elem[0]
                        x_direction += elem[1]
                if richtiger_zug:
                    return True
            return False

    def possible_felder(self, spieler):
        """
        spieler: 0 - white, 1 - black
        """
        # akt_spielfeld = self.spielfeld.copy()
        pos_felder = np.array(["empty"] * (8 * 8)).reshape((8, 8))
        counter = 0
        for row in range(8):
            for zelle in range(8):
                if self.spielfeld[row][zelle] == "empty" and self.check_rules((zelle, row), spieler, check_only=True):
                    counter += 1
                    pos_felder[row][zelle] = "pos_f"
        # self.spielfeld = akt_spielfeld.copy()
        # print("pos_felder =", counter == len(pos_felder))
        return counter, pos_felder
