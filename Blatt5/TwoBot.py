import numpy as np
import time
import matplotlib.pyplot as plt
import random


class my_Bot:
    def __init__(self, spieler_farbe):
        self.spieler = spieler_farbe == 'white'
        self.gegner_farbe = not self.spieler
        self.spielfeld = None
        self.pos_felder = None
        self.cur_choice = None
        self.timeout = False

        self.maxdepth = None
        self.checkdepth = None
        self.percentile = 1.5

        self.weights = np.array([[4, -3, 2, 2, 2, 2, -3, 4],
                                 [-3, -4, -1, -1, -1, -1, -4, -3],
                                 [2, -1, 1, 0, 0, 1, -1, 2],
                                 [2, -1, 0, 1, 1, 0, -1, 2],
                                 [2, -1, 0, 1, 1, 0, -1, 2],
                                 [2, -1, 1, 0, 0, 1, -1, 2],
                                 [-3, -4, -1, -1, -1, -1, -4, -3],
                                 [4, -3, 2, 2, 2, 2, -3, 4]])

        # Store evaluated positions - [value, value_type] indexed by [stones, player]
        self.positions = [[dict(), dict()] for i in range(64)]

    def set_next_stone(self):
        self.timeout = False
        # empty - 0, white - 1, black - 1
        position = np.zeros((8, 8))
        position[self.spielfeld == 'white'] = -1
        position[self.spielfeld == 'black'] = 1

        depth = len(np.nonzero(position)[0])
        if depth <= 6:
            self.cur_choice = random.choice(self.possible_felder(position, self.spieler, depth)[0])
            return
        deep = 2
        p = False
        while not p and not self.timeout:
            self.checkdepth = depth + deep
            self.maxdepth = depth + deep
            _, p, self.cur_choice = self.alpha_beta(position, self.spieler, depth)
            # print("len %s - %s" % ({k: len(d[0]) + len(d[1]) for k, d in enumerate(self.positions)
            #                       if len(d[0]) + len(d[1]) > 0}, self.cur_choice))
            deep += 1
        print(self.spieler, deep, p)

    def alpha_beta(self, position, spieler, depth, alpha=float('-inf'), beta=float('inf')):
        """
        position:  8x8 array with 0 - empty, -1 - white, 1 - black
        spieler:   True - White, False - Black
        depth:     # of pieces on board (depth of search)

        returns [move, pure, move]
        move: next move by calculating best end-value
              value is white number of pieces - black number of pieces
              -> white is max, black is min
        pure: True if returned value if true minmax-value or False if result of early stopping
        move: Best move to make
        """
        # Time over
        if self.timeout:
            return [float('-inf') if spieler else float('inf'), False, None]

        # Game done
        if np.count_nonzero(position) == 64:
            #  # White - # Black
            e, counts = np.unique(position, return_counts=True)
            e = list(e)
            whites = counts[e.index(-1)] if -1 in e else 0
            blacks = counts[e.index(-1)] if -1 in e else 0
            return [whites - blacks, True, None]

        # Check-depth for Prob-Cut
        # elif depth == self.checkdepth:
        #    # v >= beta with prob.of at least p? yes = > cutoff
        #    bound = round((self.percentile * sigma + beta - b) / a);
        #    if self.zero_window_alpha_beta(position, not spieler, depth, alpha=bound - 1, beta=bound) >= bound:
        #        return beta
        #    # v <= alpha with prob.of at least p? yes = > cutoff
        #   bound = round((self.percentile * sigma + alpha - b) / a);
        #  if self.zero_window_alpha_beta(position, not spieler, depth, alpha=bound, beta=bound + 1) <= bound:
        #       return alpha

        posbytes = position.tobytes()
        if posbytes in self.positions[depth][spieler]:
            value, pure, moves = self.positions[depth][spieler][posbytes]
            if pure:
                return [value, pure, moves[0] if moves is not None else None]
            if moves is None:
                print('Saved a empty move list')
                print("%s at %s / %s - %s" % (spieler, depth, self.maxdepth, moves))
                print(position, value, pure)
            mobility = len(moves)
            # Look again if unpure
            value_set = True
        else:
            pure = True
            value_set = False
            moves, mobility = self.possible_felder(position, spieler, depth)
            # TODO: Sort moves

        # Early stopping
        if depth == self.maxdepth:
            val = np.sum(np.sum(self.weights[position == (-1) ** spieler])) - \
                  np.sum(np.sum(self.weights[position == (-1) ** (not spieler)]))
            return [val * (1 + 0.1 * min(10, mobility)), False, None]

        # print("%s at %s / %s - %s" % (spieler, depth, self.maxdepth, moves))

        if len(moves) == 0:
            if self.possible_felder(position, not spieler, depth)[1] == 0:
                # Return max
                value = (-1)**(not spieler) * 64
                self.positions[depth][spieler][position.tobytes()] = [value, True, None]
                return [value, True, None]
            v, p, _ = self.alpha_beta(position, not spieler, depth, alpha=alpha, beta=beta)
            # Set alpha/beta
            if not value_set:
                value = float('-inf') if spieler else float('inf')
            if spieler is True and v > value:
                pure = pure & p
                value = v
            elif spieler is False and v < value:
                pure = pure & p
                value = v
            return [value, pure, None]

        best_id = 0
        # White
        if spieler is True:
            if not value_set:
                value = float('-inf')
            for i, move in enumerate(moves):
                # print(i, move, move[0], move[1])
                _, undos = self.check_rules(position, move[0], move[1], spieler, play=True)
                v, p, _ = self.alpha_beta(position, not spieler, depth + 1, alpha=alpha, beta=beta)
                if v > value:
                    pure = pure & p
                    value = v
                    best_id = i
                for place, undo in undos.items():
                    position[place[0], place[1]] = undo
                alpha = max(alpha, value)
                if alpha > beta:
                    break
        # Black
        else:
            if not value_set:
                value = float('inf')
            for i, move in enumerate(moves):
                # print(i, move)
                _, undos = self.check_rules(position, move[0], move[1], spieler, play=True)
                v, p, _ = self.alpha_beta(position, not spieler, depth + 1, alpha=alpha, beta=beta)
                if v < value:
                    pure = pure & p
                    value = v
                    best_id = i
                for place, undo in undos.items():
                    position[place[0], place[1]] = undo
                beta = min(beta, value)
                if alpha >= beta:
                    break

        # Reorder with best first
        moves = [moves.pop(best_id)] + moves
        self.positions[depth][spieler][position.tobytes()] = [value, pure, moves]
        try:
            return [value, pure, moves[0]]
        except TypeError:
            print("%s at %s / %s - %s" % (spieler, depth, self.maxdepth, moves))
            print(position, value, pure)
            print('alarm')

    def check_rules(self, position, row, col, spieler, play=False):
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
        undos = {(row, col): position[row, col]}

        found = False
        for stepr, stepc in [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)]:
            nrow, ncol = row + stepr, col + stepc
            flipped = False
            while nrow in range(8) and ncol in range(8):
                if not flipped:
                    # Gegnerischer Stein anliegend?
                    if position[nrow, ncol] != (-1)**(not spieler):
                        break
                    flipped = True
                else:
                    if position[nrow, ncol] == (-1)**spieler:
                        if play:
                            # Fill in
                            while nrow != row or ncol != col:
                                nrow, ncol = nrow - stepr, ncol - stepc
                                if nrow != row or ncol != col:
                                    undos.update({(nrow, ncol): position[nrow, ncol]})
                                position[nrow, ncol] = (-1)**spieler
                        found = True
                        break
                    # Lücke
                    elif position[nrow, ncol] == 0:
                        break
                nrow, ncol = nrow + stepr, ncol + stepc
        return found, undos

    def find_moves(self, position, row, col, spieler):
        """
        Same as check_rules but faster for empty boards
        """
        found = []
        for stepr, stepc in [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)]:
            nrow, ncol = row + stepr, col + stepc
            flipped = False
            while nrow in range(8) and ncol in range(8):
                if not flipped:
                    # Gegnerischer Stein anliegend?
                    if position[nrow, ncol] != (-1) ** (not spieler):
                        break
                    flipped = True
                else:
                    if position[nrow, ncol] == 0:
                        found += [(nrow, ncol)]
                        break
                    elif position[nrow, ncol] == (-1) ** spieler:
                        break
                nrow, ncol = nrow + stepr, ncol + stepc
        return found

    def possible_felder(self, position, spieler, depth):
        if depth > 42:
            # Start from empty spots
            counter = 0
            l = []
            rows, cols = np.where(position == 0)
            for i in range(len(rows)):
                row = rows[i]
                col = cols[i]
                check, _ = self.check_rules(position, row, col, spieler, play=False)
                if check:
                    l += [(row, col)]
                    counter += 1
            return l, counter
        else:
            # Start from own stones
            rows, cols = np.where(position == (-1)**spieler)
            l = list(set([m for i in range(len(rows)) for m in self.find_moves(position, rows[i], cols[i], spieler)]))
            return l, len(l)


def test1(pos, sp):
    counter = 0
    l = []
    rows, cols = np.where(pos == 0)
    for i in range(len(rows)):
        row = rows[i]
        col = cols[i]
        check, _ = b.check_rules(pos, row, col, sp, play=False)
        if check:
            l += [(row, col)]
            counter += 1
    return l


def test2(pos, sp):
    rows, cols = np.where(pos == (-1) ** sp)
    l = list(set([m for i in range(len(rows)) for m in b.find_moves(pos, rows[i], cols[i], sp)]))
    return l


if __name__ == '__main__':
    pos = np.zeros((8, 8))
    pos[3, 3] = -1
    pos[4, 4] = -1
    pos[3, 4] = 1
    pos[4, 3] = 1
    sp = True
    b = my_Bot(1)

    pl = []

    for lll in range(5, 65):
        t_ch = []
        t_fm = []
        t_t = []
        for k in range(1000):
            start = time.time()
            l = test1(pos, sp)
            t_ch += [time.time() - start]

            start = time.time()
            l = test2(pos, sp)
            t_fm += [time.time() - start]

            start = time.time()
            l, _ = b.possible_felder(pos, sp, lll)
            t_t += [time.time() - start]

        print("%d: ch: %.4f - fm: %.4f - tt: %.4f" % (lll, sum(t_ch), sum(t_fm), sum(t_t)))
        pl += [[sum(t_ch), sum(t_fm), sum(t_t)]]

        b.check_rules(pos, l[0][0], l[0][1], sp, play=True)
        sp = not sp
    print(pl)
    plt.plot(range(5, 65), [i[0] for i in pl])
    plt.plot(range(5, 65), [i[1] for i in pl])
    plt.plot(range(5, 65), [i[2] for i in pl])
    plt.show()

