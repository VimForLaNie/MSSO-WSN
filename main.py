# mssowsn_sim_clusters.py
# Simulation of "Energy efficient cluster-based routing protocol for WSN using MSSO+MST"
# Includes visualization of clustering (CN, CH, RN, BS, and routing links).
#
# ⚠️ Simplified MSSO used here (still captures the paper’s ideas).
# Parameters taken from the uploaded paper.

import math, random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

random.seed(0)
np.random.seed(0)

# --------------------------
# Paper parameters
# --------------------------
INITIAL_ENERGY = 0.5
E_ELEC = 50e-9
EPS_FS = 10e-12
EPS_MP = 0.0013e-12
EPS_DA = 5e-9
PKT_LEN = 4000
CTRL_LEN = 200
P_CH = 0.05
D0 = math.sqrt(EPS_FS / EPS_MP)

# MSSO params
MSSO_T = 30
MSSO_NALL = 30
C1, C2, C3 = 0.5, 0.05, 2.0
OMEGA = 0.5
XI = 0.5

# Simulation
MAX_ROUNDS = 1200

# --------------------------
# Energy model
# --------------------------
def dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

def etx(l_bits, d):
    if d < D0:
        return l_bits * E_ELEC + l_bits * EPS_FS * (d**2)
    else:
        return l_bits * E_ELEC + l_bits * EPS_MP * (d**4)

def erx(l_bits): return l_bits * E_ELEC
def eda(l_bits): return l_bits * EPS_DA

# --------------------------
# Node class
# --------------------------
class Node:
    def __init__(self, idx, x, y, energy=INITIAL_ENERGY):
        self.idx = idx
        self.pos = (x,y)
        self.energy = energy
        self.alive = True
        self.role = 'CN'
        self.cluster = None

# --------------------------
# MSSO-inspired optimizer (simplified)
# --------------------------
def kmeans_seed(nodes, K):
    pts = np.array([n.pos for n in nodes])
    kmeans = KMeans(n_clusters=K, n_init=5, random_state=0)
    labels = kmeans.fit_predict(pts)
    centers = kmeans.cluster_centers_
    chosen = []
    for c in centers:
        dists = [dist(n.pos, tuple(c)) for n in nodes]
        chosen.append(int(np.argmin(dists)))
    return chosen

def evaluate_CH_fitness(ch_indices, nodes, bs_pos):
    K = len(ch_indices)
    CH_nodes = [nodes[i] for i in ch_indices]
    NCH = [n for n in nodes if n.idx not in ch_indices and n.alive]
    DCHtoBS = np.mean([dist(ch.pos, bs_pos) for ch in CH_nodes])
    DNCHtoBS = np.mean([dist(n.pos, bs_pos) for n in NCH]) if NCH else 0
    # intra
    if NCH:
        sum_intra = sum(min(dist(n.pos, c.pos) for c in CH_nodes) for n in NCH)
        DIntra = sum_intra / len(NCH)
    else: DIntra = 0
    # inter
    pair_dists = [dist(CH_nodes[i].pos, CH_nodes[j].pos) 
                  for i in range(K) for j in range(i+1,K)]
    DInter = np.mean(pair_dists) if pair_dists else 0
    f1 = (DIntra + DCHtoBS) / (DInter + DNCHtoBS + 1e-12)
    mean_ErNCH = np.mean([n.energy for n in NCH]) if NCH else INITIAL_ENERGY
    mean_ErCH = np.mean([ch.energy for ch in CH_nodes]) if CH_nodes else INITIAL_ENERGY
    f2 = (mean_ErNCH+1e-12)/(mean_ErCH+1e-12)
    return OMEGA*f1 + (1-OMEGA)*f2

def MSSO_select_K(nodes, bs_pos, K):
    alive_nodes = [n for n in nodes if n.alive]
    if len(alive_nodes) < K: return [n.idx for n in alive_nodes]
    try: seed = kmeans_seed(alive_nodes, K)
    except: seed = random.sample([n.idx for n in alive_nodes], K)
    pop = []
    for i in range(MSSO_NALL):
        if i==0:
            pop.append([alive_nodes[s].idx for s in seed])
        else:
            pop.append(random.sample([n.idx for n in alive_nodes], K))
    for t in range(MSSO_T):
        fitnesses = [evaluate_CH_fitness(ind, nodes, bs_pos) for ind in pop]
        idx_sorted = np.argsort(fitnesses)
        topk = int(len(pop)/2)
        new_pop = [pop[i] for i in idx_sorted[:topk]]
        while len(new_pop) < len(pop):
            a = random.choice(new_pop)
            b = random.choice(new_pop)
            child = [random.choice((a[j], b[j])) for j in range(K)]
            while len(set(child)) < K:
                child.append(random.choice([n.idx for n in alive_nodes]))
                child = list(dict.fromkeys(child))
            new_pop.append(child[:K])
        pop = new_pop
        if t % max(1, MSSO_T//5) == 0:
            i = random.randrange(len(pop))
            ind = pop[i]
            for m in range(K//2):
                ind[m] = random.choice([n.idx for n in alive_nodes])
            pop[i] = ind
    final_fit = [evaluate_CH_fitness(ind, nodes, bs_pos) for ind in pop]
    return pop[int(np.argmin(final_fit))]

# --------------------------
# RN routing
# --------------------------
def build_intercluster_routes(rn_indices, nodes, bs_pos):
    next_hop = {}
    for i in rn_indices:
        di_bs = dist(nodes[i].pos, bs_pos)
        if di_bs < D0:
            next_hop[i] = 'BS'
            continue
        weights = {}
        for j in rn_indices:
            if j==i: continue
            dj_bs = dist(nodes[j].pos, bs_pos)
            if dj_bs >= di_bs: continue
            dij = dist(nodes[i].pos, nodes[j].pos)
            denom = (nodes[i].energy*nodes[j].energy+1e-18)
            weights[j] = etx(PKT_LEN, dij)/denom
        if not weights:
            next_hop[i] = 'BS'; continue
        jmin = min(weights, key=weights.get)
        if etx(PKT_LEN, di_bs) < (etx(PKT_LEN, dist(nodes[i].pos, nodes[jmin].pos)) + 
                                 etx(PKT_LEN, dist(nodes[jmin].pos, bs_pos))):
            next_hop[i] = 'BS'
        else:
            next_hop[i] = jmin
    return next_hop

# --------------------------
# Visualization
# --------------------------
def plot_clusters(round_num, nodes, ch_indices, rn_indices, routes, bs_pos, field_dim):
    plt.figure(figsize=(6,6))
    W,H = field_dim
    for n in nodes:
        if not n.alive: continue
        if n.role == 'CN':
            plt.scatter(*n.pos, c='gray', s=20, alpha=0.6)
    for c in ch_indices:
        if nodes[c].alive:
            plt.scatter(*nodes[c].pos, c='red', marker='*', s=150, label='CH' if c==ch_indices[0] else "")
    for r in rn_indices:
        if nodes[r].alive:
            plt.scatter(*nodes[r].pos, c='blue', marker='^', s=80, label='RN' if r==rn_indices[0] else "")
    for n in nodes:
        if n.alive and n.role=='CN' and n.cluster is not None:
            plt.plot([n.pos[0], nodes[n.cluster].pos[0]],
                     [n.pos[1], nodes[n.cluster].pos[1]], 'k--', lw=0.5, alpha=0.3)
    for ch_idx, rn_idx in zip(ch_indices, rn_indices):
        if nodes[ch_idx].alive and nodes[rn_idx].alive:
            plt.plot([nodes[ch_idx].pos[0], nodes[rn_idx].pos[0]],
                     [nodes[ch_idx].pos[1], nodes[rn_idx].pos[1]], 'g-', lw=1)
    for rn, nh in routes.items():
        if not nodes[rn].alive: continue
        if nh=='BS':
            plt.plot([nodes[rn].pos[0], bs_pos[0]],
                     [nodes[rn].pos[1], bs_pos[1]], 'b-', lw=1)
        else:
            plt.plot([nodes[rn].pos[0], nodes[nh].pos[0]],
                     [nodes[rn].pos[1], nodes[nh].pos[1]], 'b-', lw=1)
    plt.scatter(bs_pos[0], bs_pos[1], c='black', marker='s', s=120, label='BS')
    plt.xlim(0,W); plt.ylim(0,H)
    plt.title(f"Round {round_num}: Clustering")
    plt.legend()
    plt.show()

# --------------------------
# Simulation loop
# --------------------------
def run_simulation(num_nodes, field_dim, bs_pos, max_rounds=MAX_ROUNDS):
    W,H = field_dim
    nodes = [Node(i, random.uniform(0,W), random.uniform(0,H)) for i in range(num_nodes)]
    rounds=0; alive_counts=[]
    while rounds < max_rounds and any(n.alive for n in nodes):
        rounds += 1
        alive = [n for n in nodes if n.alive]
        Nalive = len(alive)
        if Nalive==0: break
        K = max(1, int(round(Nalive*P_CH)))
        ch_indices = MSSO_select_K(nodes, bs_pos, K)
        for n in nodes:
            n.role = 'CN' if n.alive else 'DEAD'
            n.cluster=None
        for c in ch_indices: nodes[c].role='CH'
        for n in nodes:
            if n.alive and n.role=='CN':
                nearest_ch = min(ch_indices, key=lambda ci: dist(n.pos,nodes[ci].pos))
                n.cluster=nearest_ch
        rn_indices=[]
        for ch in ch_indices:
            candidates=[n for n in nodes if n.alive and n.idx!=ch and n.idx not in rn_indices]
            if not candidates: rn_indices.append(ch); continue
            best=min(candidates, key=lambda v: dist(v.pos,nodes[ch].pos))
            rn_indices.append(best.idx)
        for r in rn_indices: nodes[r].role='RN'
        for n in nodes:
            if n.alive and n.role=='CN' and n.cluster is not None:
                ch=nodes[n.cluster]; d=dist(n.pos,ch.pos)
                n.energy-=etx(PKT_LEN,d); ch.energy-=erx(PKT_LEN)
                if n.energy<=0: n.alive=False
                if ch.energy<=0: ch.alive=False
        for ch_idx, rn_idx in zip(ch_indices, rn_indices):
            ch, rn=nodes[ch_idx], nodes[rn_idx]
            if not ch.alive: continue
            d=dist(ch.pos,rn.pos)
            ch.energy-=(etx(PKT_LEN,d)+eda(PKT_LEN))
            rn.energy-=erx(PKT_LEN)
            if ch.energy<=0: ch.alive=False
            if rn.energy<=0: rn.alive=False
        active_rns=[r for r in rn_indices if nodes[r].alive]
        routes=build_intercluster_routes(active_rns,nodes,bs_pos)
        for rn in active_rns:
            if not nodes[rn].alive: continue
            nh=routes.get(rn,'BS')
            if nh=='BS':
                d=dist(nodes[rn].pos,bs_pos)
                nodes[rn].energy-=etx(PKT_LEN,d)
                if nodes[rn].energy<=0: nodes[rn].alive=False
            else:
                nodes[rn].energy-=etx(PKT_LEN,dist(nodes[rn].pos,nodes[nh].pos))
                nodes[nh].energy-=erx(PKT_LEN)
                if nodes[rn].energy<=0: nodes[rn].alive=False
                if nodes[nh].energy<=0: nodes[nh].alive=False
        alive_counts.append(sum(1 for n in nodes if n.alive))
        if rounds in [1,100,500,1000]:
            plot_clusters(rounds,nodes,ch_indices,rn_indices,routes,bs_pos,field_dim)
        if all(not n.alive for n in nodes): break
    return alive_counts

# --------------------------
# Run scenarios
# --------------------------
if __name__=="__main__":
    scenarios=[{'name':'scenario1','N':100,'field':(100,100),'bs':(50,250)},
               {'name':'scenario2','N':200,'field':(200,200),'bs':(100,500)}]
    for s in scenarios:
        print("Running",s['name'])
        alive_counts=run_simulation(s['N'],s['field'],s['bs'])
        plt.plot(alive_counts,label=s['name'])
    plt.xlabel("Round"); plt.ylabel("Alive nodes")
    plt.legend(); plt.title("Alive nodes per round")
    plt.show()
