from .utilss import *
from .fasta import read_MSA
from .cluster2 import solve_cluster2


### inference functions

def Cpotts_energy(sequence, h, J, N):
    E = 0
    for i in range(N):
        E -= h[i, sequence[i]]

    for i in range(N):
        for j in range(i + 1, N):
            E -= J[i, j, sequence[i], sequence[j]]
    return E


def infer_independent_model(msa, W, save_path='none', p_prior=[]):
    N = len(msa[0])
    p = compute_weighted_averages(msa, W)

    if len(p_prior):
        p[p == 0] = p_prior[p == 0]

    if save_path != 'none':
        if os.path.exists(save_path):
            h = np.loadtxt(save_path)
            return h, np.zeros((N, N, 20, 20))

    Beff = np.sum(W)
    # (h, J) = cluster2.solve_cluster2(p, [], [], lambda_h = 0.1/Beff, lambda_J = 0)

    h = np.log(p + 0.1 / Beff)

    if save_path != 'none':
        mkdir(parentdir(save_path))
        np.savetxt(save_path, h)

    return h, np.zeros((N, N, 21, 21))


def infer_potts_ACEC(msa, contacts, W, lambda_J='auto', save_path='none', gauge='dissensus', p_prior=[]):
    N = np.shape(msa)[1]
    K = len(contacts)

    if lambda_J == 'auto':
        lambda_J = 0.01 * 20 * K * 2. / N
        # lambda_J = 10.0

    # build adjacency contacts matrix and compute selected correlations
    C = np.zeros((N, N))
    for c in contacts:
        C[c[0], c[1]] = 1
    t0 = time.time()
    av, corr, B = compute_weighted_moments_contacts(msa, W, C)

    B = float(len(msa))
    # solve 2-cluster approximation
    t0 = time.time()
    (h, J) = solve_cluster2(av, corr, contacts, lambda_h=1. / B, lambda_J=lambda_J / B, save_path=save_path)
    if params['timing']:
        print("[timing]\t ACEK minimization: %.4f" % (time.time() - t0))
    if len(p_prior):
        (h0, J0) = infer_independent_model(msa, W, p_prior=p_prior)
        h[av == 0] = h0[av == 0]

    return h, J


### service functions

def compute_weighted_averages(msa, W=None):
    if W is None:
        W = np.ones(len(msa))

    L = len(msa[0])
    columns = np.transpose(msa)
    Z = np.sum(W)
    averages = np.zeros((L, 21))

    for a in range(21):
        averages[:, a] = np.dot(columns == a, W) / Z

    return averages


def compute_weighted_moments_contacts(msa, W, C):
    # C should be an adjacency matrix
    N = np.shape(msa)[1]
    columns = np.transpose(msa)
    Z = np.sum(W)
    averages = np.zeros((N, 21))
    correlations = np.zeros((N, N, 21, 21))

    for a in range(21):
        averages[:, a] = np.dot(columns == a, W) / Z

    for i in range(N):
        for j in range(i + 1, N):
            if C[i, j]:
                for a in range(21):
                    for b in range(21):
                        correlations[i, j, a, b] = np.dot((columns[i] == a) * (columns[j] == b), W) / Z
                        correlations[j, i, b, a] = correlations[i, j, a, b]
    return [averages, correlations, Z]


def contacts_family(family, K, threshold=0):
    F = np.loadtxt('./data/%s/contacts/F.dat' % family, delimiter=',').transpose()
    sorted_F = F[:, np.argsort(-F[2, :])]
    K_contacts = []

    if threshold:
        msa = read_MSA(family)
        av = compute_weighted_averages(msa)
        idx = 0
        while len(K_contacts) < K:
            ci = int(sorted_F[0, idx] - 1)
            cj = int(sorted_F[1, idx] - 1)
            if np.sum(av[ci] == 0) + np.sum(av[cj] == 0) < threshold:
                K_contacts.append([ci, cj])
            idx += 1
    else:
        for i in range(K):
            K_contacts.append([sorted_F[0, i] - 1, sorted_F[1, i] - 1])
    return np.asarray(K_contacts, dtype=int)


def best_contacts_family(family, K):
    F = np.loadtxt('./data/%s/contacts/J_sorted.txt' % family).transpose()
    sorted_F = F[:, np.argsort(-F[3, :])]
    K_contacts = []

    for i in range(K):
        K_contacts.append([sorted_F[1, i] - 1, sorted_F[2, i] - 1])
    return np.asarray(K_contacts, dtype=int)


def real_contacts_family(family, K, threshold=0):
    F = np.loadtxt('./data/%s/contacts/atom_distances.dat' % family, delimiter=' ').transpose()
    sorted_F = F[:, np.argsort(F[2, :])]
    K_contacts = []
    if threshold:
        msa = read_MSA(family)
        av = compute_weighted_averages(msa)
        idx = 0
        while len(K_contacts) < K:
            ci = int(sorted_F[0, idx] - 1)
            cj = int(sorted_F[1, idx] - 1)
            if np.sum(av[ci] == 0) + np.sum(av[cj] == 0) < threshold:
                K_contacts.append([ci, cj])
            idx += 1
    else:
        for i in range(K):
            K_contacts.append([sorted_F[0, i] - 1, sorted_F[1, i] - 1])
    return np.asarray(K_contacts, dtype=int)


def real_contacts_family_true_positives(family, K, threshold=0):
    distances = np.loadtxt('./data/%s/contacts/atom_distances.dat' % family, delimiter=' ').transpose()
    N = int(max(distances[1]))
    print(N)
    D = np.ones((N, N)) * 1000.

    for d in distances.transpose():
        print(d)
        D[int(d[0]) - 1, int(d[1]) - 1] = d[2]
        D[int(d[1]) - 1, int(d[0]) - 1] = d[2]

    F = np.loadtxt('./data/%s/contacts/F.dat' % family, delimiter=',').transpose()
    sorted_F = F[:, np.argsort(-F[2, :])]
    K_contacts = []

    if threshold:
        msa = read_MSA(family)
        av = compute_weighted_averages(msa)
        idx = 0
        while len(K_contacts) < K:
            ci = int(sorted_F[0, idx] - 1)
            cj = int(sorted_F[1, idx] - 1)
            print(D[ci, cj])
            if (np.sum(av[ci] == 0) + np.sum(av[cj] == 0) < threshold) and (D[ci, cj] < 8.0):
                K_contacts.append([ci, cj])
            idx += 1
    else:
        for i in range(K):
            K_contacts.append([sorted_F[0, i] - 1, sorted_F[1, i] - 1])
    return np.asarray(K_contacts, dtype=int)
