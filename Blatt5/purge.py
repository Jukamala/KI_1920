import ast
import numpy as np

if __name__ == '__main__':
    with open('playbook', 'r') as b:
        dicts = ast.literal_eval(b.read())
        for depth in range(64):
            print(depth)
            for i in [0, 1]:
                dels = []
                writes = dict()
                for k, v in dicts[depth][i].items():
                    pos = np.frombuffer(k).reshape(8, 8)
                    alts = [pos, pos[:, ::-1], pos[::-1, :], pos[::-1, ::-1],
                            np.rot90(pos), np.rot90(pos)[::-1, :], np.rot90(pos)[:, ::-1], np.rot90(pos)[::-1, ::-1]]
                    values = [dicts[depth][i][a.tobytes()] for a in alts if a.tobytes() in dicts[depth][i].keys()]
                    vals = [v[0] for v in values]
                    depths = [v[2] for v in values]
                    maxdepth = max(depths)
                    if maxdepth - depth <= 2:
                        for a in [a for a in alts if a.tobytes() in dicts[depth][i].keys()]:
                            dels += [a]
                    else:
                        val = [v for j, v in enumerate(vals) if depths[j] == maxdepth]
                        moves = [tuple(v[3]) for j, v in enumerate(values) if depths[j] == maxdepth]
                        move = list(max(set(moves), key=moves.count))
                        for a in alts:
                            writes.update({a.tobytes(): [np.mean(val), False, maxdepth, move]})
                for d in list(set([d.tobytes() for d in dels])):
                    del dicts[depth][i][d]
                for w, v in writes.items():
                    dicts[depth][i][w] = v

    with open('plays', 'w') as b2:
        b2.write(str(dicts))