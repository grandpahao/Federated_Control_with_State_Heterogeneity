import os
import numpy as np
import argparse
import pickle as pkl
from tqdm import tqdm

# Definitions of a parser
parser = argparse.ArgumentParser()

## Default setting for RandomMDPs
parser.add_argument("--seed", default = 2233, type = int, help = "Choice of random seed.")
parser.add_argument("--nK", default = 5, type = int, help = "# of non-overlapped states for an agent.")
parser.add_argument("--nR", default = 5, type = int, help = "Expected # of overlapped states for an agent.")
parser.add_argument("--nN", default = 3, type = int, help = "# of agents sharing a common state.")
parser.add_argument("--N", default = 5, type = int, help = "# of agents involved.")
parser.add_argument("--gamma", default = 0.99, type = float, help = "Discounted factor for the global MDP.")
parser.add_argument("--L", default = 20000, type = int, help = "Total # of training iterations.")
parser.add_argument("--bs", default = 5, type = int, help = "Batch size for generaters in Synchronous Q-learning.")

## Default setting for ablation study on Pmax
parser.add_argument("-P","--PmaxControl", action = 'store_true', help = "Whether to carry out ablation study on Pmax.")
parser.add_argument("--pmax", default = 0.9, type = float, help = "Pmax set for different state-action pairs in one region.")

## Switching to Synchronous Q-learning
parser.add_argument("-syncQ", "--SyncQ", action = "store_true", help = "Whether to turn on synchronous sampling in Q-learning.")
parser.add_argument("-exactQ", "--ExactQ", action = "store_true", help = "Wheter to conduct exact Q-learning locally.")
parser.add_argument("--lr", default = 0.5, type = float, help = "Learning rates for synchronous Q-learning.")

args = parser.parse_args()

# Initialize the log_file
if args.PmaxControl:
    assert args.nR == 0, "Overlapped regions are not empty."
    log_dir = f"/data3/public/jinhao/FedControl/refined_record/RandomMDPs/N:{args.N},K:{args.nK},R:{args.nR},nN:{args.nN},pmax:{args.pmax}"
else:
    log_dir = f"/data3/public/jinhao/FedControl/refined_record/RandomMDPs/N:{args.N},K:{args.nK},R:{args.nR},nN:{args.nN}"
if args.SyncQ:
    log_dir += f",SyncQ,bs:{args.bs},lr:{args.lr}"
elif args.ExactQ:
    log_dir += f",ExactQ"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Auxiliary functions
## generate simplex of given shape
def softmax_simplex(shape):
    P = np.exp(np.random.uniform(size = shape))
    P = P / np.sum(P, axis=-1, keepdims = True)
    return P

## local update with aggregated Q functions
def local_update(Q, R, P, aggQ, inS):
    Vout = np.max(aggQ, axis=1)
    localV = np.max(Q, axis=1)
    Vout[inS] = localV
    tmp = P[inS] * Vout[np.newaxis, np.newaxis, :]
    maxQ = R[inS] + tmp.sum(axis=-1)*gamma
    Q = maxQ
    return Q

def local_exactQ_update(Q, R, P, aggQ, inS):
    Vout = np.max(aggQ, axis=1)
    while True:
        localV = np.max(Q, axis=1)
        Vout[inS] = localV
        tmp = P[inS] * Vout[np.newaxis, np.newaxis, :]
        maxQ = R[inS] + tmp.sum(axis=-1)*gamma
        if np.sum(np.abs(Q-maxQ)) < 1e-6:
            break
        Q = maxQ
    return Q

## local update under Synchronous Q setting
def local_syncQ_update(Q, R, P, aggQ, inS):
    K, na, ns = len(inS), P.shape[1], P.shape[2]
    Vout = np.max(aggQ, axis=1)
    localV = np.max(Q, axis=1)
    Vout[inS] = localV
    nxtV = np.zeros_like(R[inS])
    for _ in range(args.bs):
        nxtS = vector_sample(P[inS])
        nxtV += Vout[nxtS].reshape((K, na))
    maxQ = R[inS] + nxtV*gamma/args.bs
    Q = args.lr * maxQ + (1-args.lr) * Q
    return Q

def vector_sample(P):
    K, na, ns = P.shape
    flatP = P.reshape((K*na, ns))
    Rand = np.random.rand(K*na, 1)
    return np.argmax(flatP.cumsum(axis = 1)>Rand, axis=1)

## averaging strategy for FedQ
def aggregate_avg(FedQ, Sagents, Snum):
    Nshare = np.zeros(Snum)
    for S in Sagents:
        Nshare[S] += 1
    aggQ = np.zeros((Snum, nA))
    for Q, S in zip(FedQ, Sagents):
        aggQ[S] = aggQ[S] + Q/Nshare[S, np.newaxis]
    return aggQ

# Start Experiment
seed = args.seed
np.random.seed(seed)

## Loading parameters
L = args.L
nK, nR, nN, N = args.nK, args.nR, args.nN, args.N
assert nN <= N, "State is over-shared!"

### [0, nK*N) represents all non-overlapped states
### [nK*N, nS) represents all overlapped states
nS, nA = nK * N + nR * N * N // nN, 4
gamma = 0.99

### overlap_N[k] represents the idx of agents own S_{k+nK*N}
overlap_N = []
for _ in range(nR*N*N//nN):
    overlap_N.append(list(np.random.choice(np.arange(N), size=nN, p=np.ones(N)/N, replace=False)))

### Sagents[k] represents the states visible to the k-th agent
Sagents = [list(np.arange(k*nK, (k+1)*nK)) for k in range(N)]
for k in range(nS-nK*N):
    for n in overlap_N[k]:
        Sagents[n].append(k+nK*N)
        
### Generation of transition probability of SxA
P = softmax_simplex((nS,nA,nS))
if args.PmaxControl and nR == 0:
    for k in range(N):
        Sin = Sagents[k]
        Sout = [i not in Sin for i in range(nS)]
        Sout = list(np.arange(nS)[Sout])

        Pin = softmax_simplex((len(Sin), nA, len(Sin)))
        Pout = softmax_simplex((len(Sin), nA, len(Sout)))
        
        mixP = np.zeros((len(Sin), nA, nS))
        mixP[:,:,Sin], mixP[:,:,Sout] = Pin*(1-args.pmax), Pout*args.pmax
        P[Sin] = mixP

### Generation of reward function of SxA
R = np.random.rand(nS, nA)

### Statistics for Evaluation
Q_diff, Pi_diff = {}, {}

### Generation of the optimal Q function
finalQ = np.zeros((nS, nA))
while True:
    V = np.max(finalQ, axis=-1)
    tmp = P * V[np.newaxis,np.newaxis,:]
    nxtQ = R + tmp.sum(axis=-1)*gamma
    if np.sum(np.abs(finalQ-nxtQ)) < 1e-10:
        break
    finalQ = nxtQ
finalPi = np.zeros((nS, nA))
finalPi[np.arange(nS), finalQ.argmax(axis=-1)] = 1

# Control coefficient for FedQ
Es = [2, 5, 10, 20, 50]

# Baseline Q algorithm
allS = list(np.arange(nS))
globalQ = np.zeros((nS,nA))
Q_diff[1], Pi_diff[1] = [], []
for _ in tqdm(range(L)):
    if args.SyncQ:
        globalQ = local_syncQ_update(globalQ, R, P, globalQ, allS)
    elif args.ExactQ:
        globalQ = local_exactQ_update(globalQ, R, P, globalQ, allS)
    else:
        globalQ = local_update(globalQ, R, P, globalQ, allS)
    globalPi = np.zeros_like(globalQ)
    globalPi[np.arange(nS), globalQ.argmax(axis=-1)] = 1
    Q_diff[1].append(np.sum(np.abs(finalQ-globalQ)))
    Pi_diff[1].append(np.sum(np.abs(finalPi-globalPi)))

for E in Es:
    T = L // E
    
    # Federated Q algorithm
    FedQ = [np.zeros((len(Sagents[k]), nA)) for k in range(N)]
    aggQ = np.zeros((nS, nA))

    # Statistics for convergence analysis
    Q_diff[E], Pi_diff[E] = [], []

    for t in tqdm(range(T)):
        for e in range(E):
            # Updates of FedQ
            if args.SyncQ:
                nxt_FedQ = [local_syncQ_update(FedQ[k], R, P, aggQ, Sagents[k]) for k in range(N)]
            elif args.ExactQ:
                nxt_FedQ = [local_exactQ_update(FedQ[k], R, P, aggQ, Sagents[k]) for k in range(N)]
            else:
                nxt_FedQ = [local_update(FedQ[k], R, P, aggQ, Sagents[k]) for k in range(N)]
            cur_aggQ = aggregate_avg(FedQ, Sagents, nS)
            FedQ = nxt_FedQ

            # Comparison between two algorithms
            FedPi = np.zeros_like(cur_aggQ)
            FedPi[np.arange(nS), cur_aggQ.argmax(axis=-1)] = 1

            Q_diff[E].append(np.sum(np.abs(cur_aggQ-finalQ)))
            Pi_diff[E].append(np.sum(np.abs(FedPi-finalPi)))

        aggQ = cur_aggQ

log_file = os.path.join(log_dir, f"seed:{args.seed}.pkl")
with open(log_file, "wb") as f:
    data = {
        "Q": Q_diff,
        "Pi": Pi_diff
    }
    pkl.dump(data, f)

print(f"Finish {log_file}!")