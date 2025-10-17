# Full MSSO + CH/RN selection + MST inter-cluster routing simulation
# Implements the MSSO pseudocode and equations from:
# "Energy efficient cluster-based routing protocol for WSN using multi-strategy fusion snake optimizer and minimum spanning tree"
# (The code follows the equations / algorithm numbering in the paper and implements alpha-stable mutation via CMS method.)

# Dependencies:
# pip install numpy scipy matplotlib

import math
import random
import time
import copy
from datetime import datetime
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

# ---------------------------
# Simulation + MSSO settings
# ---------------------------
AREA_X, AREA_Y = 100.0, 100.0    # field
NUM_NODES = 100                 # < 100 as requested
INIT_ENERGY = 0.5                # J (table in paper)

# Radio model (Table 2 / Eqs 2-5)
E_ELEC = 50e-9           # 50 nJ/bit
EPS_FS = 10e-12          # 10 pJ/bit/m^2
EPS_MP = 0.0013e-12      # 0.0013 pJ/bit/m^4
E_DA = 5e-9              # data aggregation 5 nJ/bit
PACKET_LEN = 4000        # bits
D0 = math.sqrt(EPS_FS / EPS_MP)

BS = (50.0, 250.0)       # base station (outside field) [paper uses outside]
ROUNDS = 1200

# MSSO default parameters (paper Table 3)
NALL = 30                # population size
T_ITER = 30              # iterations inside MSSO
THQ = 0.25
THT = 0.6
C1_INIT = 0.5
C2_INIT = 0.05
C3_INIT = 2.0
THETA = 0.1
GAMMA = 1.0
DELTA = 0.0

# Fitness weights (paper used omega = phi = 0.5 in experiments)
OMEGA = 0.5      # CH fitness weight
PHI = 0.5        # RN fitness weight

# Random seed
SEED = 1991
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------
# Utility helpers
# ---------------------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def energy_tx(d, l=PACKET_LEN):
    if d < D0:
        return l * E_ELEC + l * EPS_FS * (d ** 2)
    else:
        return l * E_ELEC + l * EPS_MP * (d ** 4)

def energy_rx(l=PACKET_LEN):
    return l * E_ELEC

# ---------------------------
# Create initial nodes
# ---------------------------
def create_nodes(num_nodes=NUM_NODES):
    nodes = []
    for i in range(num_nodes):
        x = random.uniform(0, AREA_X)
        y = random.uniform(0, AREA_Y)
        nodes.append({
            'id': i,
            'pos': (x, y),
            'energy': INIT_ENERGY
        })
    return nodes

# ---------------------------
# Fuzzy C-Means (simple implementation)
# Used to seed MSSO population (paper: embed FCM centers)
# ---------------------------
def fuzzy_c_means(data, c=5, m=2.0, max_iter=100, tol=1e-5):
    # data: Nx2 numpy array
    N = data.shape[0]
    # init membership randomly
    U = np.random.dirichlet(np.ones(c), size=N)  # NxC, each row sums to 1
    for _ in range(max_iter):
        U_prev = U.copy()
        # cluster centers
        Um = U ** m
        centers = (Um.T @ data) / (Um.sum(axis=0)[:, None])
        # update U
        dist_to_centers = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)  # N x c
        # prevent zeros
        dist_to_centers = np.fmax(dist_to_centers, 1e-12)
        inv = dist_to_centers ** (-2 / (m - 1))
        U = inv / inv.sum(axis=1, keepdims=True)
        if np.linalg.norm(U - U_prev) < tol:
            break
    return centers, U

# ---------------------------
# Alpha-stable sampler (Chambers-Mallows-Stuck)
# Returns samples from S(alpha, beta=0, gamma=scale, delta=0) symmetric alpha-stable
# ---------------------------
def sample_alpha_stable(alpha, size=1, scale=1.0):
    # alpha in (0,2], beta=0 (symmetric)
    # using CMS method
    # For alpha==2, returns gaussian
    if alpha == 2.0:
        return np.random.normal(loc=0.0, scale=scale, size=size)
    elif alpha == 1.0:
        # Cauchy distribution
        return scale * np.tan(np.pi * (np.random.rand(size) - 0.5))
    else:
        U = np.random.rand(size) * math.pi - (math.pi / 2.0)   # uniform(-pi/2, pi/2)
        W = np.random.exponential(1.0, size=size)             # exponential(1)
        const = math.tan(math.pi * alpha / 2.0)
        part1 = np.sin(alpha * U) / (np.cos(U) ** (1.0 / alpha))
        part2 = (np.cos(U - alpha * U) / W) ** ((1.0 - alpha) / alpha)
        return scale * part1 * part2

# ---------------------------
# MSSO Implementation - faithful to paper
# ---------------------------
class MSSO:
    def __init__(self, nodes, K, Nall=NALL, T=T_ITER, thQ=THQ, thT=THT,
                 c1=C1_INIT, c2=C2_INIT, c3=C3_INIT,
                 theta=THETA, gamma=GAMMA, delta=DELTA,
                 embed_fcm_centers=True,
                 ch_positions=None):
        """
        nodes: list of alive node dicts (with 'pos' and 'energy')
        K: number of positions to select (e.g., CH count or RN count)
        """
        self.nodes = nodes
        self.N = len(nodes)
        self.K = K
        self.Dim = 2 * K
        self.Nall = Nall
        self.T = T
        self.thQ = thQ
        self.thT = thT
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.theta = theta
        self.gamma = gamma
        self.delta = delta
        self.embed_fcm_centers = embed_fcm_centers
        # search bounds
        self.Smin = 0.0
        self.Smax = max(AREA_X, AREA_Y)
        # positions of nodes & energies
        self.positions = np.array([n['pos'] for n in nodes])  # N x 2
        self.energies = np.array([n['energy'] for n in nodes])

        # initialize population
        self.pop = self._init_population()
        self.ch_positions = ch_positions

    def _init_population(self):
        pop = np.random.uniform(self.Smin, self.Smax, size=(self.Nall, self.Dim))
        # embed FCM cluster centers into population to accelerate convergence as paper suggests
        if self.embed_fcm_centers and self.N >= self.K:
            try:
                centers, U = fuzzy_c_means(self.positions, c=self.K, m=2.0, max_iter=100)
                # take center coordinates and place them in first individual
                coords = centers.flatten()
                if coords.size >= self.Dim:
                    pop[0, :] = coords[:self.Dim]
                else:
                    # pad with random coords
                    pad = np.random.uniform(self.Smin, self.Smax, self.Dim - coords.size)
                    pop[0, :] = np.concatenate([coords, pad])
            except Exception:
                # if FCM fails, ignore
                pass
        return pop

    def _map_individual_to_nodes(self, ind):
        # ind: 1D array of length Dim => K coordinate pairs
        coords = ind.reshape((self.K, 2))
        chosen = []
        chosen_ids = set()
        for (x, y) in coords:
            dists = np.linalg.norm(self.positions - np.array([x, y]), axis=1)
            order = np.argsort(dists)
            chosen_id = None
            for idx in order:
                if idx not in chosen_ids:
                    chosen_id = idx
                    chosen_ids.add(idx)
                    break
            if chosen_id is None:
                # fallback (shouldn't happen)
                for idx in range(self.N):
                    if idx not in chosen_ids:
                        chosen_id = idx
                        chosen_ids.add(idx)
                        break
            chosen.append(self.nodes[chosen_id])
        return chosen

    # FITNESS functions copied from paper (Eqs. 10-12 for CH, 13-19 for RN)
    def fitness_CH(self, candidate_nodes):
        K = len(candidate_nodes)
        if K == 0: return float('inf')
        # DCHtoBS (Eq.6)
        DCHtoBS = (1.0 / K) * sum(math.hypot(c['pos'][0] - BS[0], c['pos'][1] - BS[1]) for c in candidate_nodes)
        # DNCHtoBS (Eq.7)
        non_ch = [n for n in self.nodes if n not in candidate_nodes]
        if len(non_ch) == 0:
            DNCHtoBS = 0.0
        else:
            DNCHtoBS = (1.0 / len(non_ch)) * sum(math.hypot(n['pos'][0] - BS[0], n['pos'][1] - BS[1]) for n in non_ch)
        # DIntra (Eq.8) average intra-cluster: assign non-ch to nearest ch
        if len(non_ch) == 0:
            DIntra = 0.0
        else:
            per_cluster_avg = []
            for ch in candidate_nodes:
                members = [n for n in non_ch if min(dist(n['pos'], c['pos']) for c in candidate_nodes) == dist(n['pos'], ch['pos'])]
                if len(members) == 0:
                    continue
                per_cluster_avg.append(sum(dist(m['pos'], ch['pos']) for m in members) / len(members))
            DIntra = np.mean(per_cluster_avg) if len(per_cluster_avg) > 0 else 0.0
        # DInter (Eq.9)
        if K <= 1:
            DInter = 0.0
        else:
            s = 0.0; cnt = 0
            for i in range(K):
                for j in range(i+1, K):
                    s += dist(candidate_nodes[i]['pos'], candidate_nodes[j]['pos'])
                    cnt += 1
            DInter = s / cnt if cnt > 0 else 0.0
        # Eq.10 position equalization coefficient f1
        denom = (DInter + DNCHtoBS)
        f1 = (DIntra + DCHtoBS) / (denom if denom > 1e-12 else 1e-12)
        # Eq.11 energy equalization coefficient f2
        avgErNCH = np.mean([n['energy'] for n in non_ch]) if len(non_ch) > 0 else 0.0
        avgErCH = np.mean([c['energy'] for c in candidate_nodes]) if len(candidate_nodes) > 0 else 1e-12
        f2 = (avgErNCH / (avgErCH if avgErCH > 1e-12 else 1e-12))
        # Eq.12 FCH
        FCH = OMEGA * f1 + (1.0 - OMEGA) * f2
        return FCH

    def fitness_RN(self, candidate_nodes):
        K = len(candidate_nodes)
        if K == 0: return float('inf')
        # DRNtoBS as before
        DRNtoBS = (1.0 / K) * sum(math.hypot(r['pos'][0] - BS[0], r['pos'][1] - BS[1]) for r in candidate_nodes)
        # DRNtoRN as before
        if K <= 1:
            DRNtoRN = 0.0
        else:
            s = 0.0; cnt = 0
            for i in range(K):
                for j in range(i+1, K):
                    s += dist(candidate_nodes[i]['pos'], candidate_nodes[j]['pos'])
                    cnt += 1
            DRNtoRN = (2.0 / (K * (K - 1))) * s if cnt > 0 else 0.0

        if self.ch_positions is not None and len(self.ch_positions) > 0:
            DRNtoCH_vals = [min(dist(r['pos'], chpos) for chpos in self.ch_positions) for r in candidate_nodes]
            DRNtoCH = np.mean(DRNtoCH_vals)
        else:
            sorted_by_energy = sorted(self.nodes, key=lambda n: -n['energy'])
            CH_proxy = sorted_by_energy[:K] if len(sorted_by_energy) >= K else sorted_by_energy
            DRNtoCH = np.mean([min(dist(r['pos'], ch['pos']) for ch in CH_proxy) for r in candidate_nodes]) if len(CH_proxy)>0 else 0.0

        non_sel = [n for n in self.nodes if n not in candidate_nodes]
        if len(non_sel) == 0 or len(candidate_nodes) == 0:
            DCNtoCH = 1.0
        else:
            DCNtoCH = np.mean([min(dist(n['pos'], r['pos']) for r in candidate_nodes) for n in non_sel])

        g1 = (DRNtoCH + DRNtoBS + DRNtoRN) / (DCNtoCH if DCNtoCH > 1e-12 else 1e-12)
        avgErCN = np.mean([n['energy'] for n in non_sel]) if len(non_sel) > 0 else 0.0
        avgErRN = np.mean([r['energy'] for r in candidate_nodes]) if len(candidate_nodes) > 0 else 1e-12
        g2 = (avgErCN / (avgErRN if avgErRN > 1e-12 else 1e-12))
        FRN = PHI * g1 + (1.0 - PHI) * g2
        return FRN

    def run(self, select_for='CH'):
        pop = self.pop.copy()
        Nm = self.Nall // 2
        Nf = self.Nall - Nm

        best_global_mapped = None
        best_global_f = float('inf')

        for t in range(1, self.T + 1):
            rand4 = random.random()
            c1_new = C1_INIT + (1.0 / 10.0) * math.cos(rand4 * math.pi / 2.0)
            c2_new = C2_INIT + (1.0 / 1000.0) * math.cos(rand4 * math.pi / 2.0)
            c3_new = C3_INIT - 2.0 * math.sin(((t / self.T) ** 4) * math.pi / 2.0)
            self.c1, self.c2, self.c3 = c1_new, c2_new, c3_new

            Temp = math.exp(-t / self.T)
            Q = self.c1 * math.exp((t - self.T) / self.T)

            fitness_vals = np.zeros(self.Nall)
            mapped_candidates = [self._map_individual_to_nodes(pop[i]) for i in range(self.Nall)]
            for i in range(self.Nall):
                fitness_vals[i] = self.fitness_CH(mapped_candidates[i]) if select_for == 'CH' else self.fitness_RN(mapped_candidates[i])

            sorted_indices = np.argsort(fitness_vals)
            pop = pop[sorted_indices]
            fitness_vals = fitness_vals[sorted_indices]
            mapped_candidates = [mapped_candidates[i] for i in sorted_indices]

            if fitness_vals[0] < best_global_f:
                best_global_f = fitness_vals[0]
                best_global_mapped = mapped_candidates[0]

            idx_worst = self.Nall - 1
            new_pop = pop.copy()

            if Q < self.thQ: # Exploration with BDS
                Smnew_best = pop[0]
                Smworst = pop[idx_worst]
                for i in range(self.Nall):
                    r1, r2 = random.random(), random.random()
                    new_pop[i] = pop[i] + r1 * (Smnew_best - pop[i]) - r2 * (Smworst - pop[i])
            else: # Exploitation
                if Temp > self.thT: # Eat food
                    Sfood = pop[0]
                    for i in range(self.Nall):
                        randv = np.random.rand(self.Dim)
                        new_pop[i] = Sfood + self.c3 * Temp * randv * (Sfood - pop[i])
                else:
                    # GENDER-BASED LOGIC 
                    male_pop, female_pop = pop[:Nm], pop[Nm:]
                    male_fit, female_fit = fitness_vals[:Nm], fitness_vals[Nm:]

                    best_male_pos = male_pop[0] 
                    
                    # Update Males
                    for i in range(Nm):
                        if random.random() < 0.5: # Fighting
                            fi = male_fit[i] if male_fit[i] > 0 else 1e-12
                            intensity = self.c3 * math.exp(-male_fit[0] / fi)
                            new_pop[i] = male_pop[i] + intensity * random.random() * (best_male_pos - male_pop[i])
                        else: # Mating
                            partner_idx = random.randrange(Nf)
                            partner_female_pos = female_pop[partner_idx]
                            mm_i = male_fit[i] if male_fit[i] > 0 else 1e-12
                            mf_j = female_fit[partner_idx] if female_fit[partner_idx] > 0 else 1e-12
                            intensity = self.c3 * math.exp(-mf_j / mm_i)
                            new_pop[i] = male_pop[i] + intensity * random.random() * (partner_female_pos - male_pop[i])
                    
                    # Update Females
                    for i in range(Nf):
                        partner_idx = random.randrange(Nm)
                        partner_male_pos = male_pop[partner_idx]
                        mf_i = female_fit[i] if female_fit[i] > 0 else 1e-12
                        mm_j = male_fit[partner_idx] if male_fit[partner_idx] > 0 else 1e-12
                        intensity = self.c3 * math.exp(-mm_j / mf_i)
                        new_pop[i + Nm] = female_pop[i] + intensity * random.random() * (partner_male_pos - female_pop[i])

            num_replace = max(1, int(0.05 * self.Nall))
            worst_indices = np.argsort(fitness_vals)[-num_replace:]
            for wi in worst_indices:
                new_pop[wi] = np.random.uniform(self.Smin, self.Smax, size=(self.Dim,))
            
            # Adaptive alpha mutation on best male and female
            try:
                tan_arg = math.tan(t / self.T)
                alpha = 2.0 - math.exp(- (abs(tan_arg) ** 10))
            except OverflowError:
                alpha = 1.99 # Fallback for large tan
            
            if alpha > 0: # a small perturbation
                scale = 0.05
                # Mutate best male (at index 0)
                pert_male = pop[0] + pop[0] * sample_alpha_stable(alpha, size=self.Dim, scale=scale)
                mapped_pert_male = self._map_individual_to_nodes(pert_male)
                f_pert_male = self.fitness_CH(mapped_pert_male) if select_for == 'CH' else self.fitness_RN(mapped_pert_male)
                if f_pert_male < fitness_vals[0]:
                    new_pop[0] = pert_male

                # Mutate best female (at index Nm)
                if Nf > 0:
                    pert_female = pop[Nm] + pop[Nm] * sample_alpha_stable(alpha, size=self.Dim, scale=scale)
                    mapped_pert_female = self._map_individual_to_nodes(pert_female)
                    f_pert_female = self.fitness_CH(mapped_pert_female) if select_for == 'CH' else self.fitness_RN(mapped_pert_female)
                    if f_pert_female < fitness_vals[Nm]:
                        new_pop[Nm] = pert_female

            pop = np.clip(new_pop, self.Smin, self.Smax)

        return best_global_mapped if best_global_mapped is not None else self._map_individual_to_nodes(pop[0])


# ---------------------------
# Network simulation using MSSO for CH & RN
# ---------------------------
def run_simulation(nodes_init, rounds=ROUNDS, p=0.05, K_override=None, save_snap_rounds=[1,30,60,90]):
    nodes = copy.deepcopy(nodes_init)
    timeline_energy = []
    timeline_alive_nodes = [] 
    timeline_variance = [] # NEW: track variance
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if K_override is None:
        K = int(p * NUM_NODES)
    else:
        K = K_override

    for r in range(1, rounds + 1):
        print(f"\n--- Round {r} ---")
        alive = [n for n in nodes if n['energy'] > 0]
        if len(alive) < K:
            print("Not enough alive nodes for clustering.")
            remaining_rounds = rounds - len(timeline_energy)
            timeline_energy.extend([0.0] * remaining_rounds)
            timeline_alive_nodes.extend([0] * remaining_rounds)
            timeline_variance.extend([0.0] * remaining_rounds)
            break
        
        timeline_alive_nodes.append(len(alive))

        # 1) MSSO to select K CHs
        msso_ch = MSSO(alive, K, Nall=NALL, T=T_ITER, embed_fcm_centers=True)
        selected_CHs = msso_ch.run(select_for='CH')
        CH_ids = {c['id'] for c in selected_CHs}
        CHs = [n for n in nodes if n['id'] in CH_ids]

        # 2) Partition nodes into clusters
        clusters = {ch['id']: [] for ch in CHs}
        non_ch_nodes = [n for n in alive if n['id'] not in CH_ids]
        for n in non_ch_nodes:
            nearest_ch = min(CHs, key=lambda c: dist(n['pos'], c['pos']))
            clusters[nearest_ch['id']].append(n)

        # 3) For each cluster, run MSSO to pick one RN
        RNs = []
        RN_ids = set()
        for ch in CHs:
            members = [m for m in clusters.get(ch['id'], []) if m['id'] not in RN_ids]
            if not members: continue

            msso_rn = MSSO(members, K=1, Nall=max(6, min(NALL, 10)), T=max(8, T_ITER//2), embed_fcm_centers=False, ch_positions=[ch['pos']])
            chosen_rn_list = msso_rn.run(select_for='RN')
            if chosen_rn_list:
                rn_node = chosen_rn_list[0]
                if rn_node['id'] not in RN_ids:
                    RNs.append(next(n for n in nodes if n['id'] == rn_node['id']))
                    RN_ids.add(rn_node['id'])

        links = []

        # Stage 1: CN -> CH
        for n in non_ch_nodes:
            if n['id'] in RN_ids: continue
            nearest_ch = min(CHs, key=lambda c: dist(n['pos'], c['pos']))
            d = dist(n['pos'], nearest_ch['pos'])
            n['energy'] -= energy_tx(d)
            nearest_ch['energy'] -= energy_rx()
            links.append((n['pos'], nearest_ch['pos']))

        # Stage 2: CH -> RN
        for ch in CHs:
            ch['energy'] -= E_DA * PACKET_LEN
            if RNs:
                nearest_rn = min(RNs, key=lambda rn: dist(ch['pos'], rn['pos']))
                d = dist(ch['pos'], nearest_rn['pos'])
                ch['energy'] -= energy_tx(d)
                nearest_rn['energy'] -= energy_rx()
                links.append((ch['pos'], nearest_rn['pos']))
            else: 
                d_bs = dist(ch['pos'], BS)
                ch['energy'] -= energy_tx(d_bs)
                links.append((ch['pos'], BS))

        # Stage 3: RN -> BS (via MST)
        active_RNs = [rn for rn in RNs if rn['energy'] > 0]
        if len(active_RNs) > 1:
            pos = np.array([rn['pos'] for rn in active_RNs])
            D = distance_matrix(pos, pos)
            visited, edges = [0], []
            while len(visited) < len(active_RNs):
                min_d, pair = float('inf'), None
                for i in visited:
                    for j in range(len(active_RNs)):
                        if j not in visited and D[i, j] < min_d:
                            min_d, pair = D[i, j], (i, j)
                if pair is None: break
                edges.append(pair)
                visited.append(pair[1])
            
            for (i, j) in edges:
                a, b = active_RNs[i], active_RNs[j]
                d = dist(a['pos'], b['pos'])
                a['energy'] -= energy_tx(d)
                b['energy'] -= energy_rx()
                links.append((a['pos'], b['pos']))
            
            rn_to_bs = min(active_RNs, key=lambda x: dist(x['pos'], BS))
            rn_to_bs['energy'] -= energy_tx(dist(rn_to_bs['pos'], BS))
            links.append((rn_to_bs['pos'], BS))
        elif len(active_RNs) == 1:
            rn = active_RNs[0]
            rn['energy'] -= energy_tx(dist(rn['pos'], BS))
            links.append((rn['pos'], BS))

        total_energy = sum(max(0.0, n['energy']) for n in nodes)
        timeline_energy.append(total_energy)
        
        # NEW: Calculate and store variance
        alive_energies = [n['energy'] for n in nodes if n['energy'] > 0]
        if len(alive_energies) > 1:
            variance = np.var(alive_energies)
            timeline_variance.append(variance)
        else:
            timeline_variance.append(0.0) # Variance is 0 if 1 or 0 nodes are alive

        if r in save_snap_rounds:
            fname = f"mss o_snapshot_round{r}_{timestamp}.png".replace(" ", "_")
            visualize_snapshot(nodes, CHs, RNs, links, title=f"MSSO clusters at round {r}", fname=fname)

    return timeline_energy, timeline_alive_nodes, timeline_variance

# ---------------------------
# Visualization helpers
# ---------------------------
def visualize_snapshot(nodes, CHs, RNs, links, title="snapshot", fname=None):
    plt.figure(figsize=(8, 8))
    for a, b in links:
        plt.plot([a[0], b[0]], [a[1], b[1]], linestyle='--', color='gray', linewidth=0.8, alpha=0.7)
    
    ch_ids = {c['id'] for c in CHs}
    rn_ids = {r['id'] for r in RNs}
    
    handles, labels = plt.gca().get_legend_handles_labels()
    
    for n in nodes:
        if n['energy'] <= 0: continue
        label_map = {'CH': 'red', 'RN': 'blue', 'CN': 'gray'}
        
        if n['id'] in ch_ids:
            label = 'CH'
            marker='^'; s=90
        elif n['id'] in rn_ids:
            label = 'RN'
            marker='s'; s=70
        else:
            label = 'CN'
            marker='o'; s=30

        if label not in labels:
            plt.scatter(n['pos'][0], n['pos'][1], marker=marker, c=label_map[label], s=s, label=label)
            labels.append(label)
        else:
            plt.scatter(n['pos'][0], n['pos'][1], marker=marker, c=label_map[label], s=s)

    if 'BS' not in labels:
        plt.scatter(BS[0], BS[1], marker='*', c='green', s=160, label='BS')
    else:
        plt.scatter(BS[0], BS[1], marker='*', c='green', s=160)
        
    plt.title(title)
    margin = 10
    plt.xlim(-margin, AREA_X + margin)
    plt.ylim(-margin, BS[1] + margin)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300)
        print(f"Saved snapshot: {fname}")
    plt.close()

# ---------------------------
# LEACH baseline (for comparison)
# ---------------------------
def simulate_leach_baseline(nodes_init, rounds=ROUNDS, p=0.05, save_first_round=False, timestamp=None):
    nodes = copy.deepcopy(nodes_init)
    timeline_energy = []
    timeline_alive_nodes = []
    timeline_variance = [] # NEW: track variance
    
    for r in range(1, rounds + 1):
        alive = [n for n in nodes if n['energy'] > 0]
        timeline_alive_nodes.append(len(alive)) 
        
        if not alive:
            remaining_rounds = rounds - len(timeline_energy)
            timeline_energy.extend([0.0] * remaining_rounds)
            timeline_alive_nodes.extend([0] * (remaining_rounds+1)) 
            timeline_alive_nodes = timeline_alive_nodes[:rounds]
            timeline_variance.extend([0.0] * remaining_rounds)
            break
            
        K = max(1, int(p * len(alive)))
        
        CHs = random.sample(alive, K)
        CH_ids = {c['id'] for c in CHs}
        links = []
        
        non_ch_nodes = [n for n in alive if n['id'] not in CH_ids]
        for n in non_ch_nodes:
            nearest_ch = min(CHs, key=lambda c: dist(n['pos'], c['pos']))
            d = dist(n['pos'], nearest_ch['pos'])
            n['energy'] -= energy_tx(d)
            nearest_ch['energy'] -= energy_rx()
            links.append((n['pos'], nearest_ch['pos']))
        
        for ch in CHs:
            ch['energy'] -= E_DA * PACKET_LEN
            ch['energy'] -= energy_tx(dist(ch['pos'], BS))
            links.append((ch['pos'], BS))
            
        timeline_energy.append(sum(max(0.0, n['energy']) for n in nodes))
        
        # NEW: Calculate and store variance
        alive_energies = [n['energy'] for n in nodes if n['energy'] > 0]
        if len(alive_energies) > 1:
            variance = np.var(alive_energies)
            timeline_variance.append(variance)
        else:
            timeline_variance.append(0.0)

        if save_first_round and r == 1 and timestamp:
            fname = f"leach_snapshot_round1_{timestamp}.png"
            visualize_snapshot(nodes, CHs, [], links, title="LEACH round 1", fname=fname)
            
    return timeline_energy, timeline_alive_nodes, timeline_variance

# ---------------------------
# Top-level run & saving figs
# ---------------------------
def main():
    nodes = create_nodes(NUM_NODES)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Running LEACH baseline ...")
    energy_leach, alive_leach, variance_leach = simulate_leach_baseline(nodes, rounds=ROUNDS, p=0.05, save_first_round=True, timestamp=timestamp)

    print(f"\nRunning MSSO optimized protocol...")
    energy_msso, alive_msso, variance_msso = run_simulation(nodes, rounds=ROUNDS, p=0.1, save_snap_rounds=[1, 30, 60, 90])

    # Plot 1: Energy Comparison
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, 1 + len(energy_leach)), energy_leach, 'r-', label='LEACH (baseline)')
    plt.plot(range(1, 1 + len(energy_msso)), energy_msso, 'b-', label='MSSO-MST optimized')
    plt.xlabel('Round')
    plt.ylabel('Total remaining energy (J)')
    plt.title('Energy comparison: LEACH vs MSSO-MST optimized')
    plt.grid(True)
    plt.legend()
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tight_layout()
    out_name = f"energy_comparison_{timestamp}.png"
    plt.savefig(out_name, dpi=300)
    print(f"\nSaved energy comparison figure: {out_name}")
    plt.close()

    # Plot 2: Alive Nodes Comparison
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, 1 + len(alive_leach)), alive_leach, 'r-', label='LEACH (baseline)')
    plt.plot(range(1, 1 + len(alive_msso)), alive_msso, 'b-', label='MSSO-MST optimized')
    plt.xlabel('Round')
    plt.ylabel('Number of Alive Nodes')
    plt.title('Alive Nodes Comparison: LEACH vs MSSO-MST optimized')
    plt.grid(True)
    plt.legend()
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tight_layout()
    out_name_alive = f"alive_nodes_comparison_{timestamp}.png"
    plt.savefig(out_name_alive, dpi=300)
    print(f"Saved alive nodes comparison figure: {out_name_alive}")
    plt.close()

    # Plot 3: Variance Comparison (NEW)
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, 1 + len(variance_leach)), variance_leach, 'r-', label='LEACH (baseline)')
    plt.plot(range(1, 1 + len(variance_msso)), variance_msso, 'b-', label='MSSO-MST optimized')
    plt.xlabel('Round')
    plt.ylabel('Variance of Node Energy Consumption')
    plt.title('Network Energy Consumption Variance')
    plt.grid(True)
    plt.legend()
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tight_layout()
    out_name_variance = f"variance_comparison_{timestamp}.png"
    plt.savefig(out_name_variance, dpi=300)
    print(f"Saved variance comparison figure: {out_name_variance}")
    plt.close()

if __name__ == "__main__":
    main()
