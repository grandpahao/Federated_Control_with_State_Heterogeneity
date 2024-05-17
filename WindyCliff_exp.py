import os
import numpy as np
import argparse
import pickle as pkl

# Definitions of a parser
parser = argparse.ArgumentParser()

## Default setting for RandomMDPs
# parser.add_argument("--seed", default = 2233, type = int, help = "Choice of random seed.")
parser.add_argument("--nRow", default = 5, type = int, help = "# of rows in WindyCliffs.")
parser.add_argument("--nCol", default = 5, type = int, help = "# of columns in WindyCliffs.")
# parser.add_argument("--nN", default = 3, type = int, help = "# of agents sharing a common state.")
parser.add_argument("--N", default = 5, type = int, help = "# of agents involved.")
parser.add_argument("--gamma", default = 0.99, type = float, help = "Discounted factor for the global MDP.")
parser.add_argument("--L", default = 50000, type = int, help = "Total # of training iterations.")
parser.add_argument("--split_dir", default = "v", choices = ["v", "h"], help = "How to split state space for N involved agents.")
parser.add_argument("--windp", default = 0.2, type = float, help = "Power of wind.")

args = parser.parse_args()

lr, bs = 0.1, 10

# Construct the environment of WindyCliffs
def WindyCliff(M, N, nAgent, split = "v", wind_dir = 2, wind_p = 0.2):
    # Agents get stuck in cliff_region, cliff_regions are absorbing states.
    pos = np.arange(0, M*N)
    cliff_region = np.arange((M-1)*N+1, M*N-1)
    land_region = np.concatenate([np.arange(0,(M-1)*N+1), np.arange(M*N-1, M*N)])
    
    # There are four actions {right, left, down, up}.
    # With probability of wind_p, agents will take the action of {wind_dir} no matter its choice of action.
    # windP models the transition dynamic of action corresponding to {wind_dir}.
    P, windP = np.zeros((M*N, 4, M*N)), np.zeros((M*N, 4, M*N))
    dx, dy = [0, 0, 1, -1], [1, -1, 0, 0]
    for a in range(4):
        for p in land_region:
            x, y = p//N, p%N
            wind_nx, wind_ny = np.clip(x+dx[wind_dir], 0, M-1), np.clip(y+dy[wind_dir], 0, N-1)
            nx, ny = np.clip(x+dx[a], 0, M-1), np.clip(y+dy[a], 0, N-1)
            P[p][a][nx*N+ny] = 1
            windP[p][a][wind_nx*N+wind_ny] = 1
        for p in cliff_region:
            P[p][a][p] = 1
            windP[p][a][p] = 1
    P = wind_p * windP + (1-wind_p) * P
    
    # cliff_region will get punished, while getting to the end is rewarded.
    R = -0.01 * np.ones((M*N, 4)) # agents are encouraged to move quickly.
    R[cliff_region, :] = -0.1*np.ones((cliff_region.shape[0], 4))
    R[M*N-1, :] = 1.0*np.ones(4)
    
    Sagents = []
    if split == "v":
        wL = N // nAgent
        for k in range(nAgent):
            if k == nAgent-1:
                Sagent = list(np.arange(k*wL, N))
                for n in range(1, M):
                    Sagent += list(np.arange(k*wL, N)+N*n)
            else:
                Sagent = list(np.arange(k*wL, (k+1)*wL))
                for n in range(1, M):
                    Sagent += list(np.arange(k*wL, (k+1)*wL)+N*n)
            Sagents.append(Sagent)
    elif split == "h":
        wL = M // nAgent
        for k in range(nAgent):
            if k == nAgent -1:
                Sagent = list(np.arange(k*wL*N, M*N))
            else:
                Sagent = list(np.arange(k*wL*N, min((k+1)*wL, M)*N))
            Sagents.append(Sagent)
    else:
        raise NotImplementedError("Unknown splitting directions!")
    return P, R, Sagents

# local update with aggregated Q functions
def local_update(Q, R, P, aggQ, inS):
    Vout = np.max(aggQ, axis=1)
    localV = np.max(Q, axis=1)
    Vout[inS] = localV
    tmp = P[inS] * Vout[np.newaxis, np.newaxis, :]
    maxQ = R[inS] + tmp.sum(axis=-1)*gamma
    Q = maxQ
    return Q

def local_syncQ_update(Q, R, P, aggQ, inS):
    K, na, ns = len(inS), P.shape[1], P.shape[2]
    Vout = np.max(aggQ, axis=1)
    localV = np.max(Q, axis=1)
    Vout[inS] = localV
    nxtV = np.zeros_like(R[inS])
    for _ in range(bs):
        nxtS = vector_sample(P[inS])
        nxtV += Vout[nxtS].reshape((K, na))
    maxQ = R[inS] + nxtV*gamma/bs
    Q = lr * maxQ + (1-lr) * Q
    return Q

def vector_sample(P):
    K, nA, nS = P.shape
    flatP = P.reshape((K*nA, nS))
    Rand = np.random.rand(K*nA, 1)
    return np.argmax(flatP.cumsum(axis = 1)>Rand, axis=1)

# averaging strategy for FedQ
def aggregate_avg(FedQ, Sagents, Snum):
    Nshare = np.zeros(Snum)
    for S in Sagents:
        Nshare[S] += 1
    aggQ = np.zeros((Snum, nA))
    for Q, S in zip(FedQ, Sagents):
        aggQ[S] = aggQ[S] + Q/Nshare[S, np.newaxis]
    return aggQ

gamma, nA = 0.99, 4
windps = [0.1, 0.3, 0.5, 0.7, 0.9]

m, n, N, split = args.nRow, args.nCol, args.N, args.split_dir

Q_diff = {}
log_dir = f"/data3/public/jinhao/FedControl/refined_record/WindyCliffs/m:{m},n:{n},N:{N},split:{split},SyncQ,lr:{lr},bs:{bs}"
os.makedirs(log_dir)
for p in windps:
    
    log_file = os.path.join(log_dir, f"windp:{p}.pkl")
    
    P, R, Sagents = WindyCliff(M = m, N = n, nAgent = N, split = split, wind_p = p)
    nS, nAgent = m*n, N
    
    finalQ = np.zeros((nS, nA))
    while True:
        V = np.max(finalQ, axis=-1)
        tmp = P * V[np.newaxis,np.newaxis,:]
        nxtQ = R + tmp.sum(axis=-1)*gamma
        if np.sum(np.abs(finalQ-nxtQ)) < 1e-20:
            break
        finalQ = nxtQ

    # Control coefficient for FedQ
    Es = [2, 5, 10, 20, 50]
    L = 20000

    # Baseline Q algorithm
    allS = list(np.arange(nS))
    globalQ = np.zeros((nS,nA))
    Q_diff[1] = []
    for _ in range(L):
        globalQ = local_syncQ_update(globalQ, R, P, globalQ, allS)
        Q_diff[1].append(np.sum(np.abs(finalQ-globalQ)))

    for E in Es:
        T = L // E

        # Federated Q algorithm
        FedQ = [np.zeros((len(Sagents[k]), nA)) for k in range(nAgent)]
        aggQ = np.zeros((nS, nA))

        # Statistics for convergence analysis
        Q_diff[E] = [] 

        for t in range(T):
            for e in range(E):
                # Updates of FedQ
                nxt_FedQ = [local_syncQ_update(FedQ[k], R, P, aggQ, Sagents[k]) for k in range(nAgent)]
                cur_aggQ = aggregate_avg(FedQ, Sagents, nS)
                FedQ = nxt_FedQ

                Q_diff[E].append(np.sum(np.abs(cur_aggQ-finalQ)))

            aggQ = cur_aggQ
    
    with open(log_file, "wb") as f:
        data = {"Q": Q_diff}
        pkl.dump(data, f)
    print(f"Finish {log_file}!")