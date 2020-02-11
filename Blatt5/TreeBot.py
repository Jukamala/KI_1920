import numpy as np
import time
import matplotlib.pyplot as plt
import random


def stable(pos, spieler):
    # Stable
    mask = pos != (-1) ** spieler
    mask2 = mask[:, ::-1]
    left = np.where(mask.any(axis=1), mask.argmax(axis=1), 8)
    right = np.where(mask2.any(axis=1), mask2.argmax(axis=1), 8)

    ld = np.array([left[0]] + 7 * [0])
    lu = np.array(7 * [0] + [left[7]])
    rd = np.array([right[0]] + 7 * [0])
    ru = np.array(7 * [0] + [right[7]])
    done = [left[0] > 0, left[7] > 0, right[0] > 0, right[7] > 0]
    for i in range(7):
        if done[0]:
            if left[i + 1] > 0:
                ld[i + 1] = min(ld[i], left[i + 1])
            else:
                done[0] = False
        if done[1]:
            if left[6 - i] > 0:
                lu[6 - i] = min(lu[7 - i], left[6 - i])
            else:
                done[1] = False
        if done[2]:
            if right[i + 1] > 0:
                rd[i + 1] = min(rd[i], right[i + 1])
            else:
                done[2] = False
        if done[3]:
            if right[6 - i] > 0:
                ru[6 - i] = min(ru[7 - i], right[6 - i])
            else:
                done[3] = False
    return np.sum(np.minimum(np.maximum(lu, ld) + np.maximum(ru, rd), 8))

class my_Bot:
    def __init__(self, spieler_farbe):
        self.spieler = spieler_farbe == 'white'
        self.gegner_farbe = not self.spieler
        self.spielfeld = None
        self.pos_felder = None
        self.cur_choice = None
        self.timeout = False
        self.maxdepth = None

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
        while not p:
            self.maxdepth = depth + deep
            try:
                v, p, self.cur_choice = self.alpha_beta(position, self.spieler, depth)
            except ResourceWarning:
                break
            deep += 1

        print("%s searched till depth %d %s value %d" %
              (["Black", "White"][self.spieler], deep-1, ["estimating", "achieving"][p], v))

        # Prepare next round
        self.timeout = False
        self.positions = [[dict(), dict()] for i in range(64)]

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
            raise ResourceWarning('Timeout')

        # Game done
        if np.count_nonzero(position) == 64:
            #  # White - # Black
            e, counts = np.unique(position, return_counts=True)
            e = list(e)
            whites = counts[e.index(-1)] if -1 in e else 0
            blacks = counts[e.index(1)] if 1 in e else 0
            return [1000 * (whites - blacks), True, None]

        posbytes = position.tobytes()
        if posbytes in self.positions[depth][spieler]:
            value, pure, moves = self.positions[depth][spieler][posbytes]
            if pure:
                return [value, pure, moves[0] if moves is not None else None]
            pure = True
            mobility = len(moves)
        else:
            pure = True
            moves, mobility = self.possible_felder(position, spieler, depth)

        # Early stopping
        if depth == self.maxdepth:
            # Importance of heuristics
            k1, k2, k3 =\
                [[-2, 6, 25], [-2, 5, 20], [0, 4, 15], [2, 3, 8], [3, 3, 8], [4.5, 1.5, 8], [6, 0, 6]][depth//10]

            # Mobility
            _, enemy_mobility = self.possible_felder(position, not spieler, depth)
            mob = (-1)**(not spieler) * (mobility - enemy_mobility)

            # Difference in stones
            val = np.sum(np.sum(position == (-1) ** spieler)) - np.sum(np.sum(position == (-1) ** (not spieler)))

            # Bonus for stables
            white_stables = stable(position, True)
            black_stables = stable(position, False)
            if white_stables > 32:
                self.positions[depth][spieler][posbytes] = [value, True, moves]
                return [white_stables * 1000, True, None]
            if black_stables > 32:
                self.positions[depth][spieler][posbytes] = [value, True, moves]
                return [-black_stables * 1000, True, None]
            stab = (white_stables - black_stables)
            return [k1 * val + k2 * mobility + k3 * stab, False, None]

        # print("%s at %s / %s - %s" % (spieler, depth, self.maxdepth, moves))

        if len(moves) == 0:
            if self.possible_felder(position, not spieler, depth)[1] == 0:
                # Return max
                val = np.sum(np.sum(position == -1)) - np.sum(np.sum(position == 1))
                value = (-1) ** (val < 0) * 65000 if val != 0 else 0
                self.positions[depth][spieler][posbytes] = [value, True, None]
                return [value, True, None]
            v, p, _ = self.alpha_beta(position, not spieler, depth, alpha=alpha, beta=beta)
            # Set alpha/beta
            value = float('-inf') if spieler else float('inf')
            if spieler is True and v > value:
                pure = pure & p
                value = v
            elif spieler is False and v < value:
                pure = pure & p
                value = v
            return [value, pure, None]

        best_ids = []
        # White
        if spieler is True:
            value = float('-inf')
            for i, move in enumerate(moves):
                # print(i, move)
                _, undos = self.check_rules(position, move[0], move[1], spieler, play=True)
                v, p, _ = self.alpha_beta(position, not spieler, depth + 1, alpha=alpha, beta=beta)
                if v > value:
                    pure = pure & p
                    value = v
                    best_ids = [i] + best_ids
                for place, undo in undos.items():
                    position[place[0], place[1]] = undo
                alpha = max(alpha, value)
                if alpha > beta:
                    break
        # Black
        else:
            value = float('inf')
            for i, move in enumerate(moves):
                # print(i, move)
                _, undos = self.check_rules(position, move[0], move[1], spieler, play=True)
                v, p, _ = self.alpha_beta(position, not spieler, depth + 1, alpha=alpha, beta=beta)
                if v < value:
                    pure = pure & p
                    value = v
                    best_ids = [i] + best_ids
                for place, undo in undos.items():
                    position[place[0], place[1]] = undo
                beta = min(beta, value)
                if alpha >= beta:
                    break

        # Reorder with best first
        moves = [moves[i] for i in best_ids] + [m for i, m in enumerate(moves) if i not in best_ids]
        self.positions[depth][spieler][position.tobytes()] = [value, pure, moves]
        return [value, pure, moves[0]]

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
                    if position[nrow, ncol] != (-1) ** (not spieler):
                        break
                    flipped = True
                else:
                    if position[nrow, ncol] == (-1) ** spieler:
                        if play:
                            # Fill in
                            while nrow != row or ncol != col:
                                nrow, ncol = nrow - stepr, ncol - stepc
                                if nrow != row or ncol != col:
                                    undos.update({(nrow, ncol): position[nrow, ncol]})
                                position[nrow, ncol] = (-1) ** spieler
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
            rows, cols = np.where(position == (-1) ** spieler)
            l = list(set([m for i in range(len(rows)) for m in self.find_moves(position, rows[i], cols[i], spieler)]))
            return l, len(l)