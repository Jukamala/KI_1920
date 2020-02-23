import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np


def a_star(g, h, s, t, animate=False, pos=None, title=None):
    """
    Returns the nodes of a shortest path from s to t in graph g using A* with
    costs c as edge attribute and a monotone estimator h.
    Can be optionally animated.
    """
    if animate:
        fig, ax = plt.subplots()
        if title is not None:
            fig.canvas.set_window_title(title)
        if pos is None:
            pos = nx.spring_layout(g)
    searched = []
    search_queue = {s: h[s]}
    cost = {n: float('inf') for n in g.nodes()}
    cost[s] = 0
    pre = {n: None for n in g.nodes()}
    time = dict()
    i = 0
    print('                g   h   f')
    while len(search_queue) > 0:
        # Get nearest node
        n = min(search_queue, key=search_queue.get)
        print('(%1d) Knoten: %s, %3d %3d %3d - searched:%s, queue:%s' %
              (i, n, cost[n], h[n], cost[n] + h[n], searched, search_queue))
        del search_queue[n]
        # Keep track of When we visit what
        time[n] = i
        i += 1
        if animate:
            animate_a_star(g, pos, ax, time, cost, h)
        # Is end reached?
        if n == t:
            break
        # Update Neighbors
        for p in g[n]:
            if p not in searched:
                new_cost = cost[n] + g[n][p]['c']
                if new_cost < cost[p]:
                    cost[p] = new_cost
                    pre[p] = n
                    search_queue[p] = new_cost + h[p]
        searched += [n]
    if pre[t] is None:
        raise ValueError('No path from %s to %s' % (s, t))
    path = [t]
    while path[0] != s:
        path = [pre[path[0]]] + path
    animate_a_star(g, pos, ax, time, cost, h, path=path)
    plt.show()
    return path


def animate_a_star(g, pos, ax, time, cost, h, path=None):
    ax.clear()
    edge_colors = 'grey' if path is None else\
        ['darkorange' if (u, v) in zip(path[:-1] + path[1:], path[1:] + path[:-1]) else 'grey' for u, v in g.edges()]
    node_labels = {n: (n if n not in time.keys() else "%s (%d)\n%3d\n%3d\n%3d" % (n, time[n], cost[n], h[n], cost[n] + h[n]))
                   for n in g.nodes()}
    nx.draw(g, pos=pos, ax=ax, node_size=800, node_shape="s", linewidths=4, width=2, alpha=0.9,
            node_color='skyblue', edge_color=edge_colors)
    nx.draw_networkx_labels(g, labels=node_labels, pos=pos, font_size=7, font_color="dimgrey", font_weight="bold")
    nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=nx.get_edge_attributes(g, 'c'), ax=ax, font_size=8, alpha=0.9)
    plt.pause(0.5)


def bps(g, s, t):
    # Breitensuche in kreisfreien Graphen
    queue = [s]
    pre = {s: None}

    while len(queue) > 0:
        n = queue.pop()
        if n == t:
            break
        for p in g[n]:
            if p != pre[n]:
                pre[p] = n
                queue = [p] + queue
    if t not in pre.keys():
        raise ValueError('No path from %s to %s' % (s, t))
    path = [t]
    while path[0] is not None:
        path = [pre[path[0]]] + path

    return path


def dps(g, s, t, pre=None):
    # Tiefensuche in kreisfreien Graphen
    if s == t:
        return [s]
    neig = [n for n in g[s] if n != pre]
    if len(neig) == 0:
        return []

    for p in neig:
        l = dps(g, p, t, pre=s)
        if len(l) > 0:
            return [s] + l
    if pre is None:
        raise ValueError('No path from %s to %s' % (s, t))
    return []


def minmax(g, node, max_player, alpha=float('-inf'), beta=float('inf')):
    # Min-Max with Alpha-Beta-Prunning
    if g.nodes[node]['val'] is not None:
        return g.nodes[node]['val']
    if max_player:
        value = float('-inf')
        for n in g[node]:
            value = max(value, minmax(g, n, False, alpha=alpha, beta=beta))
            alpha = max(alpha, value)
            if alpha >= beta:
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


def tree_layout(g, s):
    # Tree Layout for directed graph without loops
    # Group by Layers
    grouped = [[s]]
    i = 0
    while len(grouped) > i:
        next_group = [x for i in grouped[i] for x in list(g[i])]
        i += 1
        if len(next_group) > 0:
            grouped += [next_group]
    # Compute pos
    y_steps = 1 / (len(grouped) - 1)
    pos = dict()
    for i, group in enumerate(grouped):
        y = 1 - i * y_steps
        x_list = list(np.linspace(0, 1, len(group) + 2))[1:-1]
        pos.update({n: (x_list[j], y) for j, n in enumerate(group)})
    return pos


def aufgabe_2_1():
    g = nx.Graph()
    g.add_nodes_from("abcdefgst")
    g.add_edges_from([('a', 'e', {'c': 9}), ('a', 's', {'c': 53}), ('b', 'd', {'c': 67}), ('b', 'e', {'c': 32}),
                      ('b', 's', {'c': 29}), ('c', 'd', {'c': 24}), ('c', 's', {'c': 12}), ('e', 'f', {'c': 58}),
                      ('e', 'g', {'c': 37}), ('f', 'g', {'c': 14}), ('f', 't', {'c': 40}), ('g', 't', {'c': 28})])
    h = {'a': 75, 'b': 104, 'c': 139, 'd': 160, 'e': 90, 'f': 37, 'g': 17, 't': 0, 's': 128}

    # 2.1.a)
    pos = nx.spring_layout(g)
    path = a_star(g, h, 's', 't', animate=True, pos=pos, title='A* für Aufgabe 2.1.a)')
    print("Shortest Path from s to t is %s\n" % path)

    # 2.1.b) After the change of the heuristic it isn't monotone anymore: h(f) = 37 > 17 = 3 + 14 = h(g) + c(f,g)
    # (for h to be monotone it has to hold: h(x) <= h(y) + c(x,y) for all x,y

    # 2.1.c) We can't use the non-monotone h so we'll use the old one:
    # We start from s because it isn't more work compared to starting in e
    g['g']['t']['c'] = 23
    path = a_star(g, h, 's', 't', animate=True, pos=pos, title='A* für Aufgabe 2.1.c)')
    print("Shortest Path from s to t is %s\n" % path)


def aufgabe_2_2():
    g = nx.DiGraph()
    g.add_nodes_from(range(28))
    g.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (1, 5), (2, 6), (2, 7), (2, 8), (2, 9),
                      (3, 10), (3, 11), (4, 12), (4, 13), (5, 14), (5, 15), (5, 16), (5, 17),
                      (6, 18), (6, 19), (6, 20), (7, 21), (7, 22), (7, 23), (8, 24), (8, 25),
                      (9, 26), (9, 27)])
    val = {x: y for x, y in zip(range(10, 28), [7,15,20,23,16,3,2,1,6,17,10,3,1,2])}
    """
    g.add_nodes_from(range(33))
    g.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 6), (2, 7), (3, 8), (3, 9), (4, 10),
                      (4, 11), (5, 12), (6, 13), (6, 14), (7, 15), (8, 16), (9, 17), (9, 18), (10, 19),
                      (10, 20), (11, 21), (11, 22), (11, 23), (12, 24), (13, 25), (14, 26), (14, 27),
                      (15, 28), (16, 29), (17, 30), (17, 31), (18, 32)])
    val = {x: y for x, y in zip(range(19, 33), [7, 9, 1, 2, 8, 6, 5, 6, 9, 7, 5, 9, 8, 6])}
    """
    nx.set_node_attributes(g, None, 'val')
    nx.set_node_attributes(g, None, 'alpha')
    nx.set_node_attributes(g, None, 'beta')
    nx.set_node_attributes(g, val, 'val')
    minmax(g, 0, max_player=True)

    fig, ax = plt.subplots()
    fig.canvas.set_window_title('Alpha-Beta-Prunning für Aufgabe 2.2)')
    pos = tree_layout(g, 0)
    prettify = lambda x: '' if x is None else (r'$\infty$' if x == float('inf') else
                                               (r'-$\infty$' if x == float('-inf') else x))
    node_labels = {n: data['val'] or r"$\alpha$= %s" % prettify(data['alpha']) + "\n" +
                                     r"$\beta$= %s" % prettify(data['beta']) for n, data in g.nodes(data=True)}
    nx.draw(g, pos=pos, ax=ax, node_size=800, node_shape="s", linewidths=4, width=2, alpha=0.9,
            node_color='skyblue', edge_color='grey')
    nx.draw_networkx_labels(g, labels=node_labels, pos=pos, font_size=9, font_color="dimgrey", font_weight="bold")
    plt.show()


def aufgabe_2_3():
    # Create Labyrinth
    g = nx.grid_2d_graph(15, 15)
    pos = {elem: elem for i, elem in enumerate(sorted(g.nodes()))}
    while len(g.edges()) > 15 ** 2 - 1:
        r = random.choice(list(g.edges()))
        g.remove_edge(*r)
        if not nx.is_connected(g):
            g.add_edge(*r)

    # Breitensuche
    path = bps(g, (12, 0), (0, 14))

    fig, ax = plt.subplots()
    fig.canvas.set_window_title('Labyrinth für Aufgabe 2.3)')
    edge_colors = ['darkorange' if (u, v) in zip(path[:-1] + path[1:], path[1:] + path[:-1])
                   else 'grey' for u, v in g.edges()]
    nx.draw(g, pos=pos, ax=ax, node_size=50, linewidths=4, width=2, alpha=0.9,
            node_color='skyblue', edge_color=edge_colors)
    nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=nx.get_edge_attributes(g, 'c'), ax=ax, font_size=8, alpha=0.9)
    plt.show()


if __name__ == '__main__':
    # aufgabe_2_1()
    aufgabe_2_2()
    # aufgabe_2_3()
