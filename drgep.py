#Direct interaction coefficients

import heapq
import numpy as np

def species_fluxes(net, R):
    ns = net.n_species
    F = np.zeros(ns) 
    D = np.zeros(ns)
    nu = []
    for j,rx in enumerate(net.reactions):
        d = {}
        keys = set(rx.r_counts) | set(rx.p_counts)
        for i in keys:
            i0 = i - 1                      # 1-based file index -> 0-based array
            cr = rx.r_counts.get(i,0)
            cp = rx.p_counts.get(i,0)
            F[i0] += cp * R[j]
            D[i0] += cr * R[j]
            d[i0] = (cp+cr)
        nu.append(d)
    G = np.maximum(F, D)
    return F, D, G, nu

def direct_interaction(net, R, G, nu):
    num = {}
    for j, rx in enumerate(net.reactions):
        if R[j] == 0.0:
            continue
        parts = list(nu[j].keys())
        for A in parts:
            c = abs(nu[j][A]) * R[j]
            if c == 0.0:
                continue
            row = num.setdefault(A, {})
            for B in parts:
                if B == A:
                    continue
                row[B] = row.get(B, 0.0) + c
    r = {}
    for A, row in num.items():
        if G[A] <= 0.0:
            continue
        r[A] = {B: min(v/G[A],1.0) for B, v in row.items() if v>0.0}
    return r

def drgep_scores(net, r, targets):
    ns = net.n_species
    best = np.zeros(ns)
    for t in targets:
        dist = np.full(ns, np.inf)
        dist[t] = 0.0
        pq = [(0.0, t)]
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
                    heapq.heappush(pq, (nd, v))
        best = np.maximum(best, np.exp(-dist))
    for t in targets:
        best[t] = 1.0
    return best

def brute_force_scores(net, r, target, max_depth=4):
    #for toy networks
    ns = net.n_species
    best = np.zeros(ns)
    best[target] = 1.0

    def walk(u, val, visited, depth):
        if depth == 0:
            return
        for v, w in r.get(u, {}).items():
            if v in visited:
                continue
            nv = val * w
            if nv > best[v]:
                best[v] = nv
            walk(v, nv, visited | {v}, depth - 1)

    walk(target, 1.0, {target}, max_depth)
    return best

def reduced_network(net, scores, eps):
    #Species with R_A >= eps; reactions whose species are ALL retained
    keep_sp = set(np.where(scores >= eps)[0])
    keep_rx = [j for j, rx in enumerate(net.reactions)
               if all((i - 1) in keep_sp for i in rx.reactants + rx.products)]
    return keep_sp, keep_rx
