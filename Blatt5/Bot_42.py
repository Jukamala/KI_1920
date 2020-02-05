import numpy as np
import time


class my_Bot:
    def __init__(self, spieler_farbe):
        self.spieler_farbe = spieler_farbe == 'white'
        self.gegner_farbe = not self.spieler_farbe
        self.spielfeld = None
        self.pos_felder = None
        self.cur_choice = None
        self.timeout = False

        # Store evaluated positions - [value, value_type]
        self.positions = dict()

    def set_next_stone(self):
        # empty - 0, white - 1, black - 1
        position = np.zeros((8, 8))
        position[self.spielfeld == 'white'] = -1
        position[self.spielfeld == 'black'] = 1

        for depth in range(2, 8):
            _, _, self.cur_choice = self.alpha_beta(position, self.spieler_farbe, 1, depth)
            print("len %s" % len(self.positions))
            self.positions = dict()

    def alpha_beta(self, position, spieler, depth, maxdepth, alpha=float('-inf'), beta=float('inf')):
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
        # Game done
        if np.count_nonzero(position) == 64:
            #  # White - # Black
            e, counts = np.unique(position, return_counts=True)
            e = list(e)
            whites = counts[e.index(-1)] if -1 in e else 0
            blacks = counts[e.index(-1)] if -1 in e else 0
            return [whites - blacks, True, None]

        # Early stopping
        elif depth == maxdepth:
            # Use heuristic
            e, counts = np.unique(position, return_counts=True)
            e = list(e)
            whites = counts[e.index(-1)] if -1 in e else 0
            blacks = counts[e.index(1)] if 1 in e else 0
            return [whites - blacks, False, None]

        posbytes = position.tobytes()
        if posbytes in self.positions:
            value, pure, moves = self.positions[posbytes]
            if pure:
                return [value, pure, moves[0]]
            # Look again for unpure
            value_set = True
        else:
            pure = True
            value_set = False
            moves, mobility = self.possible_felder(position, spieler)
            # TODO: Sort moves

        # print("%s at %s / %s - %s" % (spieler, depth, maxdepth, moves))

        if len(moves) == 0:
            # TODO Two dicts
            if len(self.possible_felder(position, not spieler)) == 0:
                # Return max
                value = (-1)**(not spieler) * 64
                self.positions[position.tobytes()] = [value, True, None]
                return [value, True, []]
            v, p, _ = self.alpha_beta(position, spieler, depth + 1, maxdepth, alpha=alpha, beta=beta)
            # Set alpha/beta
            if not value_set:
                value = float('-inf') if spieler else float('inf')
            if spieler is True and v > value:
                pure = pure & p
                value = v
                alpha = max(alpha, value)
            elif spieler is False and v < value:
                pure = pure & p
                value = v
                beta = min(beta, value)
            return [value, pure, []]

        best_id = 0
        # White
        if spieler is True:
            if not value_set:
                value = float('-inf')
            for i, move in enumerate(moves):
                # print(i, move, move[0], move[1])
                _, undos = self.check_rules(position, move[0], move[1], spieler, play=True)
                v, p, _ = self.alpha_beta(position, not spieler, depth + 1, maxdepth, alpha=alpha, beta=beta)
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
                v, p, _ = self.alpha_beta(position, not spieler, depth + 1, maxdepth, alpha=alpha, beta=beta)
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
        self.positions[position.tobytes()] = [value, pure, moves]
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

    def possible_felder(self, position, spieler):
        val, counts = np.unique(position, return_counts=True)
        ind = list(val)
        stones = counts[ind.index((-1)**spieler)] if (-1)**spieler in ind else 0
        holes = counts[ind.index(0)] if 0 in ind else 0

        if stones > holes:
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
