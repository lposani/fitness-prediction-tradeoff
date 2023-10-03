from .imports import *
from .utilss import *


def fasta_to_msa(filename):
    parent = parentdir(filename)
    cmd = 'bash convert_fasta.sh ' + filename + ' > ' + parent + '/sequences.msa'
    os.system(cmd)
    return parent + '/sequences.msa'


def encode_with_gap(s):
    alphabet = 'ACDEFGHIKLMNPQRSTVWY-'
    return alphabet[s]


def decode_with_gap(a):
    dict = {
        'A': 0, 'a': 0,
        'C': 1, 'c': 1,
        'D': 2, 'd': 2,
        'E': 3, 'e': 3,
        'F': 4, 'f': 4,
        'G': 5, 'g': 5,
        'H': 6, 'h': 6,
        'I': 7, 'i': 7,
        'K': 8, 'k': 8,
        'L': 9, 'l': 9,
        'M': 10, 'm': 10,
        'N': 11, 'n': 11,
        'P': 12, 'p': 12,
        'Q': 13, 'q': 13,
        'R': 14, 'r': 14,
        'S': 15, 's': 15,
        'T': 16, 't': 16,
        'V': 17, 'v': 17,
        'W': 18, 'w': 18,
        'Y': 19, 'y': 19,
        '-': 20, 'B': 11, 'Z': 13
    }
    return dict[a]


def decode_sequence_with_gap(seq):
    num = np.array([], dtype=int)
    for s in seq:
        num = np.append(num, decode_with_gap(s))
    return num


def encode_sequence_with_gap(seq):
    str = ""
    for s in seq:
        str = str + encode_with_gap(s)
    return str


def converter_with_gap(string):
    sequence = np.zeros(len(string), dtype=int)
    for i in range(len(string)):
        sequence[i] = decode_with_gap(string[i])
    return sequence


def read_MSA(family):
    msa = np.load('./data/%s/sequences.msa.npy' % family)
    return msa


def read_WT(family):
    filename = './data/%s/wt.faa' % family
    f = open(filename)
    data = [d[:-1] for d in f.readlines()]
    wt_letters = data[1]
    return decode_sequence_with_gap(wt_letters)


def read_mutations(family):
    savename = './data/%s/delta_fitness.txt' % family
    if os.path.exists(savename) == 0:
        filename = './data/%s/fitness_mutations.faa' % family
        f = open(filename)
        data = [d[:-1] for d in f.readlines()]
        ntot = int(len(data) / 2)
        delta_fitness = []
        mutated_sequences = []
        for i in range(ntot):
            delta_fitness.append(float(data[2 * i][1:]))
            mutated_sequences.append(decode_sequence_with_gap(data[2 * i + 1]))
        np.savetxt(savename, delta_fitness)
        np.savetxt("./data/%s/mutated_sequences.txt" % family, mutated_sequences, fmt='%d')
        return np.asarray(mutated_sequences), np.asarray(delta_fitness)
    else:
        delta_fitness = np.loadtxt(savename)
        mutated_sequences = np.loadtxt("./data/%s/mutated_sequences.txt" % family, dtype=int)
        return mutated_sequences, delta_fitness


def read_double_mutations(family):
    wt = read_WT(family)
    filename = './data/%s/fitness_double_mutations.faa' % family
    f = open(filename)
    data = [d[:-1] for d in f.readlines()]
    ntot = len(data) / 2
    mutated_site_1 = np.zeros(ntot)
    mutated_site_2 = np.zeros(ntot)
    delta_fitness = []
    mutated_sequences = []
    for i in range(ntot):
        delta_fitness.append(float(data[2 * i][1:]))
        mutated_sequences.append(decode_sequence_with_gap(data[2 * i + 1]))

    return np.asarray(mutated_sequences), np.asarray(delta_fitness)


def compute_reweighting(family, max_homology):
    fname = './data/%s/weights/reweighting_%.3f.dat' % (family, max_homology)

    if os.path.exists(fname):
        return np.loadtxt(fname)

    msa = read_MSA(family)
    if max_homology == 0:
        return np.ones(len(msa))

    N = float(len(msa[0]))
    weights = np.zeros(len(msa))

    for i, s in enumerate(msa):
        n_neighbors = np.sum(np.sum(s != msa, 1) < max_homology * N)
        weights[i] = 1. / n_neighbors
        # print('---', i, '---', n_neighbors)

    mkparent(fname)
    np.savetxt(fname, weights)
    return weights


def compute_weights_MSA(msa, max_homology):
    # there has to be a faster way to do this
    if max_homology == 0:
        return np.ones(len(msa))
    N = float(len(msa[0]))
    weights = np.zeros(len(msa))

    for i, s in enumerate(msa):
        n_neighbors = np.sum(np.sum(s != msa, 1) < max_homology * N)
        weights[i] = 1. / n_neighbors
        # print('---', i, '---', n_neighbors)
    return weights


def main_distance_matrix(msa):
    (B, N) = np.shape(msa)
    D = np.zeros((B, B))
    for i, s in enumerate(msa):
        D[i] = np.sum(s != msa, 1) / float(N)
    return D


def histogram_distance(family, ax='auto', reweighting=0):
    autoflag = 0
    msa = read_MSA(family)
    if reweighting:
        W = np.loadtxt('./data/%s/weights/reweighting_%.3f.dat' % (family, reweighting))
    else:
        W = np.ones(len(msa))
    wt = read_WT(family)
    distances = np.sum(wt != msa, 1)
    N = len(wt)
    if ax == 'auto':
        autoflag = 1
        f, axs = plt.subplots(1, 2, figsize=(8, 4))
        ax = axs[0]
        axs[1].hist(distances, range(N), weights=W, density=True)
        axs[1].set_xlabel('Hamming distance to WT')
        axs[1].set_ylabel('count')
        axs[1].set_yscale('log')
    ax.hist(distances, range(N))
    ax.set_xlabel('Hamming distance to WT')
    ax.set_ylabel('count')
    ax.set_title(family)
    if autoflag:
        print("autosaving")
        plt.savefig('./data/%s/distance_hist.pdf' % family)
        plt.close()


def B_N(family):
    msa = read_MSA(family)
    B, N = msa.shape
    return B, N


def is_mutated(family):
    savename = './data/%s/is_mutated.txt' % family
    if os.path.exists(savename)==0:
        mutated_sequences, deltafitness = read_mutations(family)
        N = len(mutated_sequences[0])
        wt = read_WT(family)

        mutated = np.zeros((N, 21))
        for i in range(N):
            for a in range(21):
                mutated[i, a] = np.sum(np.transpose(mutated_sequences)[i] == a)
        np.savetxt(savename, mutated)
    else:
        mutated = np.loadtxt(savename)
    return mutated


def subsample_msa(family, distance, B, msa='none', precision = 0.01, **kwargs):
    wt = read_WT(family)
    N = float(len(wt))
    if msa=='none':
        msa=read_MSA(family)

    reweighting = 0
    given_D = 0
    for key, value in kwargs.items():
        if key=='reweighting' and value:
            reweighting = value
        if key=='main_distance_matrix':
            main_D = value
            given_D = True

            # W = np.loadtxt('../Data/%s/weights/reweighting_%.3f.dat' % (family, reweighting))

    dists = np.sum(wt != msa, 1)/N
    MSAs = []
    Ws = []
    distances = []

    # assess unbiased distance
    d_unbiased = np.mean(dists)

    for A in [1, 10, 100, 1000, 10000]:
        for K in [11, 21, 51, 101]:
            if distance > d_unbiased:
                alphas = np.linspace(-A, 0, K)
            else:
                alphas = np.linspace(A, 0, K)

            for alpha in alphas:
                probs = np.exp(-alpha*dists)
                probs = probs/np.sum(probs)
                try:
                    subsampled_index = np.random.choice(range(len(msa)), B, replace=False, p=probs)
                except:
                    continue

                subsampled_msa = msa[subsampled_index]

                if reweighting:
                    if given_D:
                        # t0=time.time()
                        sub_D = main_D[subsampled_index, :]
                        sub_D = sub_D[:, subsampled_index]
                        neighbors = np.sum(sub_D < reweighting, 1)
                        subsampled_W = 1./neighbors
                        # dt = time.time()-t0
                        # t0 = time.time()
                        # subsampled_W = fasta.compute_weights_MSA(subsampled_msa, reweighting)
                        # print dt, "-- vs --", time.time() - t0
                    else:
                        subsampled_W = compute_weights_MSA(subsampled_msa, reweighting) #W[subsampled_index]
                else:
                    subsampled_W = np.ones(len(subsampled_msa))
                D_eff = np.average(dists[subsampled_index], weights=subsampled_W)

                #print '%.3f'%alpha, D_eff

                if np.abs(D_eff - distance) < precision:
                    if reweighting:
                        return subsampled_msa, D_eff, subsampled_W
                    else:
                        return subsampled_msa, D_eff
                else:
                    MSAs.append(subsampled_msa)
                    distances.append(D_eff)
                    Ws.append(subsampled_W)

                if distance > d_unbiased:
                    if D_eff < distance - 0.01:
                        break # no point in looking at larger exponents
                if distance < d_unbiased:
                    if D_eff > distance + 0.01: # no point in looking at smaller exponents
                        break

    distances = np.asarray(distances)
    best_idx = np.argmin((distance - distances)**2.)
    best_msa = MSAs[best_idx]
    best_D = distances[best_idx]
    best_W = Ws[best_idx]
    if params['debug']:
        print("[subsample_msa]\t [!] sampled msa under precision - %s B = %u: desired D = %.3f, best D = %.3f" % (
        family, B, distance, best_D))
    if reweighting:
        return best_msa, best_D, best_W
    else:
        return best_msa, best_D, np.ones(len(best_msa))

