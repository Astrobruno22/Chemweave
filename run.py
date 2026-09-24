#Driver - run DRGEP over the disk model and build the graph

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from Network import Network
from model import DiskModel
from rate import RateEvaluator
import drgep

RATEFILE = "ratefile"
MODELDIR = "model"
XION = "xionrates.txt"
DUST = "dustdensity"

TARGETS = ["CO", "H2O", "OH", "HCO+", "H2", "E-", "C+"]

def setup():
    #Everything needed for one run
    net = Network(RATEFILE)
    dm = DiskModel(MODELDIR, net, XION, DUST)
    ev = RateEvaluator(net)
    return net, dm, ev

def rates(net, dm, ev, c):
    #k and R for one cell
    st = dm.cell(c)
    k = ev.k(st["T"], st["density"], st["G0"], st["Av"], st["xion"], net.cr, st["gr"])
    R = ev.fluxes(k, st["n"])
    return st, k, R

def score_cell(net, dm, ev, c, targets):
    #DRGEP importance for every species in one cell
    st, k, R = rates(net, dm, ev, c)
    F, D, G, nu = drgep.species_fluxes(net, R)
    r = drgep.direct_interaction(net, R, G, nu)
    s = drgep.drgep_scores(net, r, targets)
    return s, G, r

def run_all(out="scores.npy"):
    #Score every cell, save the raw scores - do NOT threshold here
    net, dm, ev = setup()
    targets = [net.index_of[t] - 1 for t in TARGETS]
    S = np.zeros((dm.ncells, net.n_species))
    Gall = np.zeros((dm.ncells, net.n_species))
    for c in range(dm.ncells):
        S[c], Gall[c], _ = score_cell(net, dm, ev, c, targets)
        if c % 1000 == 0:
            print("cell", c, "of", dm.ncells)
    np.save(out, S)
    np.save("G.npy", Gall)
    print("saved", out, S.shape)
    return S, Gall

#__graph__

def build_graph(net, r, scores):
    #Full DRGEP graph for one cell. Nodes = species, edges = r_AB
    g = nx.DiGraph()
    for i in range(net.n_species):
        g.add_node(net.name_of[i], score=scores[i])
    for A, row in r.items():
        for B, w in row.items():
            g.add_edge(net.name_of[A], net.name_of[B], weight=w)
    return g

def path_tree(net, r, target):
    #Only the edges Dijkstra used. This is what the score actually means,
    #and it is readable - the full graph is not
    ns = net.n_species
    dist = np.full(ns, np.inf)
    parent = np.full(ns, -1, int)
    dist[target] = 0.0
    import heapq
    pq = [(0.0, target)]
    seen = np.zeros(ns, bool)
    while pq:
        d, u = heapq.heappop(pq)
        if seen[u]:
            continue
        seen[u] = True
        for v, w in r.get(u, {}).items():
            if w <= 0.0:
                continue
            nd = d - np.log(w)
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    g = nx.DiGraph()
    for v in range(ns):
        if parent[v] >= 0:
            a = int(parent[v])
            g.add_edge(net.name_of[a], net.name_of[v], weight=r[a][v])
    return g, np.exp(-dist)

def draw(g, scores_by_name, target, title="", fname=None, top=25):
    #Keep the top-scoring nodes only, otherwise it is a hairball
    keep = sorted(g.nodes, key=lambda n: -scores_by_name.get(n, 0.0))[:top]
    if target not in keep:
        keep.append(target)
    h = g.subgraph(keep)
    pos = nx.spring_layout(h, seed=3, k=1.6, iterations=400)
    size = [200 + 2600 * scores_by_name.get(n, 0.0) for n in h.nodes]
    col = ["#c94f2f" if n == target else "#3b6ea5" for n in h.nodes]
    wid = [0.4 + 3.5 * h[u][v]["weight"] for u, v in h.edges]
    plt.figure(figsize=(11, 8))
    nx.draw_networkx_edges(h, pos, width=wid, edge_color="#888", arrowsize=11, node_size=size)
    nx.draw_networkx_nodes(h, pos, node_size=size, node_color=col, alpha=0.9)
    lab = {n: (x, y - 0.075) for n, (x, y) in pos.items()}
    nx.draw_networkx_labels(h, lab, font_size=8,
                            bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.15))
    plt.title(title, fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=140)
    plt.close()

def graph_for_cell(c=1157, target="CO", fname="graph.png"):
    #Build and draw the path tree for one cell
    net, dm, ev = setup()
    targets = [net.index_of[t] - 1 for t in TARGETS]
    s, G, r = score_cell(net, dm, ev, c, targets)
    t = net.index_of[target] - 1
    g, sc = path_tree(net, r, t)
    by_name = {net.name_of[i]: sc[i] for i in range(net.n_species)}
    st = dm.cell(c)
    title = ("DRGEP path tree from %s, cell %d\nAv=%.2e  T=%.0f K  nH=%.2e"
             % (target, c, st["Av"], st["T"], st["density"]))
    draw(g, by_name, target, title, fname)
    print("wrote", fname)
    return g, sc

#__diagnostics__

def balance(net, dm, ev, c):
    #F should equal D at steady state. This is the only ground truth we have
    st, k, R = rates(net, dm, ev, c)
    F, D, G, nu = drgep.species_fluxes(net, R)
    return np.abs(F - D).sum() / G.sum() if G.sum() > 0 else np.nan

def census(net, S, G, eps=1e-3):
    #Split the deletions by cause - they mean different things
    zero = (G <= 0)
    unreach = (G > 0) & (S == 0.0)
    below = (G > 0) & (S > 0.0) & (S < eps)
    kept = (S >= eps)
    print("eps =", eps)
    print("  kept         ", kept.sum(1).mean().round(1), "species per cell")
    print("  below eps    ", below.sum(1).mean().round(1))
    print("  unreachable  ", unreach.sum(1).mean().round(1))
    print("  zero flux    ", zero.sum(1).mean().round(1))
    print("  zero flux everywhere:",
          [net.name_of[i] for i in range(net.n_species) if zero[:, i].all()])
    return dict(zero=zero, unreach=unreach, below=below, kept=kept)

if __name__ == "__main__":
    net, dm, ev = setup()
    print(net.n_species, "species,", net.n_reactions, "reactions")
    print("dead reactions:", len(ev.dead_j))
    print("balance, cell 1157: %.3f" % balance(net, dm, ev, 1157))
    graph_for_cell(1157, "CO", "graph.png")
    S, G = run_all()
    census(net, S, G, 1e-3)
