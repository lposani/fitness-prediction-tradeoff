from .imports import *
from scipy.optimize import minimize
from .utilss import mkdir

def cross_entropy(x, pi, pj, P, lambda_h, lambda_J):
    qi = len(pi)
    qj = len(pj)
    J = np.reshape(x[qi + qj :], (qi, qj))
    hi = x[:qi]
    hj = x[qi:qi+qj]

    # compute partition function
    J_plus_hihj = J + np.tile(hi, (qj,1)).transpose() + np.tile(hj, (qi,1))
    Z = 1 + np.sum(np.exp(hi)) + np.sum(np.exp(hj)) + np.sum(np.exp(J_plus_hihj))

    # compute data -log-likelihood
    E = np.sum(J * P) + np.sum(hi * pi) + np.sum(hj * pj)

    # compute regularization
    R = lambda_h * (np.sum(hi * hi) + np.sum(hj * hj)) + lambda_J * np.sum(J * J)

    return np.log(Z) - E + R


def gradient(x, pi, pj, P, lambda_h, lambda_J):
    G = np.zeros(len(x))
    qi = len(pi)
    qj = len(pj)
    J = np.reshape(x[qi + qj :], (qi, qj))
    hi = x[:qi]
    hj = x[qi:qi+qj]

    # compute partition function
    J_plus_hihj = J + np.tile(hi, (qj, 1)).transpose() + np.tile(hj, (qi, 1))
    Z = 1 + np.sum(np.exp(hi)) + np.sum(np.exp(hj)) + np.sum(np.exp(J_plus_hihj))

    # compute gradient
    J_plus_hi = J + np.tile(hi, (qj, 1)).transpose()
    J_plus_hj = J + np.tile(hj, (qi,1))

    G[:qi] = (np.exp(hi) * (1 + np.sum(np.exp(J_plus_hj),1)))/Z - pi + 2*lambda_h * hi
    G[qi:(qi+qj)] = (np.exp(hj) * (1 + np.sum(np.exp(J_plus_hi),0)))/Z - pj + 2*lambda_h * hj
    G[qi + qj:] = (np.exp(J_plus_hihj) / Z - P + 2 * lambda_J * J).flatten()

    return G


def cross_entropy_ind(x, pi, lambda_h):
    hi = x
    # compute partition function
    Z = 1 + np.sum(np.exp(hi))

    # compute data -log-likelihood
    E = np.sum(hi * pi)

    # compute regularization
    R = lambda_h * np.sum(hi * hi)

    return np.log(Z) - E + R


def gradient_ind(x, pi, lambda_h):
    hi = x

    # compute partition function
    Z = 1 + np.sum(np.exp(hi))

    # compute gradient
    G = np.exp(hi)/Z - pi + 2*lambda_h * hi

    return G


def solve_independent(pi, lambda_h):
    # define minimanda
    f = lambda x: cross_entropy_ind(x, pi, lambda_h)
    g = lambda x: gradient_ind(x, pi, lambda_h)
    x0 = np.zeros(len(pi))
    res = minimize(f, x0, method='BFGS', jac=g, options={'disp': False})
    hi = res.x
    return hi


def solve_cluster2_ij(pi, pj, P, lambda_h, lambda_J):
    # prepare parameter inputs
    qi = len(pi)
    qj = len(pj)

    J0 = np.zeros((qi, qj))
    hi0 = np.zeros(qi)
    hj0 = np.zeros(qj)

    x0 = np.concatenate((hi0, hj0, J0.flatten()))

    # define minimanda
    f = lambda x: cross_entropy(x, pi, pj, P, lambda_h, lambda_J)
    g = lambda x: gradient(x, pi, pj, P, lambda_h, lambda_J)

    # minimize!
    # t0 = time.time()
    res = minimize(f, x0, method='BFGS', jac=g, options={'disp': False, 'gtol' : 0.0005})
    # print '==========', time.time() - t0
    hi = res.x[:qi]
    hj = res.x[qi:qi+qj]
    J = np.reshape(res.x[qi+qj:], (qi, qj))

    return hi, hj, J


def solve_cluster2(av, corr, contacts, lambda_h, lambda_J, gauge='dissensus', p_cut=0.0, save_path='none'):
    N = np.shape(av)[0]
    A = np.shape(av)[1]
    J = np.zeros((N, N, A, A))

    # TODO add cluster 1 minimization
    # h = np.zeros((N, A))
    # h_ind = np.zeros((N, A))
    # for i in range(N):
    #     h_sol = solve_cluster1_i(av[i], lambda_h)
    #     h[i] = h_sol
    #     h_ind[i] = h_sol

    h = np.zeros((N, A))
    h_ind = np.zeros((N, A))

    for i in range(N):
        vi, gi = gauge_vector(av[i], gauge=gauge)
        h_sol = solve_independent(vi, lambda_h)
        h_sol = ungauge_vector(h_sol, gi)
        h[i] = h_sol
        h_ind[i] = h_sol
        unseen_index_i = np.asarray(av[i] == 0, dtype=bool)
        dissensus_prob_i = np.min(av[i][av[i] > 0])
        h[i][unseen_index_i] = np.log(lambda_h / dissensus_prob_i)

    # write cross entropy for independent model to find the starting point
    for c in contacts:
        i = c[0]
        j = c[1]

        # print "[cluster2.py] --- solving 2-cluster inference for couple (%u, %u)" % (i, j)

        vi = av[i]
        vj = av[j]
        corrij = corr[i,j]

        # # compress colors with p_cut
        vi, where_i, compressed_p_i = compress_vector(vi, p_cut)
        vj, where_j, compressed_p_j = compress_vector(vj, p_cut)
        corrij = compress_matrix(corrij, where_i, where_j)

        # # gauge average vectors and correlation matrix
        vi, gi = gauge_vector(vi, gauge=gauge)
        vj, gj = gauge_vector(vj, gauge=gauge)

        corrij = gauge_matrix(corrij, gi, gj)
        # solve 2-cluster problem

        if save_path=='none':
            hi, hj, Jij = solve_cluster2_ij(vi, vj, corrij, lambda_h, lambda_J)

        if save_path!='none' and os.path.exists(save_path+'/hi_%u_%u_lJ=%.6f.dat' % (i,j, lambda_J)) == 0:
            mkdir(save_path)
            hi, hj, Jij = solve_cluster2_ij(vi, vj, corrij, lambda_h, lambda_J)
            np.savetxt(save_path+'/hi_%u_%u_lJ=%.6f.dat' % (i,j, lambda_J), hi)
            np.savetxt(save_path + '/hj_%u_%u_lJ=%.6f.dat' % (i,j, lambda_J), hj)
            np.savetxt(save_path + '/Jij_%u_%u_lJ=%.6f.dat' % (i,j, lambda_J), Jij)
        if save_path != 'none' and os.path.exists(save_path + '/hi_%u_%u_lJ=%.6f.dat' % (i,j, lambda_J)) == 1:
            hi = np.loadtxt(save_path + '/hi_%u_%u_lJ=%.6f.dat' % (i,j, lambda_J))
            hj = np.loadtxt(save_path + '/hj_%u_%u_lJ=%.6f.dat' % (i,j, lambda_J))
            Jij = np.loadtxt(save_path + '/Jij_%u_%u_lJ=%.6f.dat' % (i,j, lambda_J))

        # # # un-gauge fields and couplings
        hi = ungauge_vector(hi, gi)
        hj = ungauge_vector(hj, gj)
        Jij = ungauge_matrix(Jij, gi, gj)

        # un-compress colors
        dissensus_prob_i = np.min(av[i][av[i]>0])
        ind_pseudocount_i = lambda_h/dissensus_prob_i
        hi = uncompress_vector(hi, where_i, compressed_p_i, ind_pseudocount_i)

        dissensus_prob_j = np.min(av[j][av[j] > 0])
        ind_pseudocount_j = lambda_h / dissensus_prob_j
        hj = uncompress_vector(hj, where_j, compressed_p_j, ind_pseudocount_j)

        Jij = uncompress_matrix(Jij, where_i, where_j)

        # iterate on estimation
        deltahi = hi - h_ind[i]
        deltahj = hj - h_ind[j]

        J[i,j] = Jij
        J[j,i] = Jij.transpose()

        h[i] += deltahi
        h[j] += deltahj

    for i in range(N):
        unseen_index_i = np.asarray(av[i] == 0, dtype=bool)
        dissensus_prob_i = np.min(av[i][av[i] > 0])
        # print unseen_index_i
        h[i][unseen_index_i] = np.log(lambda_h / dissensus_prob_i)
        # h[unseen_index_i] = h[i, gi] + np.log(lambda_h/av[i, gi])
        # print h[i, gi] + np.log(lambda_h/av[i, gi])
        #
        # unseen_index_j = np.asarray(av[j] == 0, dtype=bool)
        # h[unseen_index_j] = h[j, gj] + np.log(lambda_h / av[j, gj])

    return h,J


# gauge

def gauge_vector(v, gauge='dissensus'):
    if gauge == 'consensus':
        g = np.argmax(v)
    if gauge == 'dissensus':
        non_zero_idx = np.where(v > 0)[0] # the smaller but not zero
        g = non_zero_idx[v[non_zero_idx].argmin()]
    v_g = np.delete(v, g)
    return v_g, g


def ungauge_vector(v, gi):
    return np.insert(v, gi, 0)


def gauge_matrix(corr, gi, gj):
    corr_g_i = np.delete(corr, gi, axis=0)
    corr_g_ij = np.delete(corr_g_i, gj, axis=1)
    return corr_g_ij


def ungauge_matrix(corr, gi, gj):
    qi = np.shape(corr)[0]
    qj = np.shape(corr)[1]
    corr_ui = np.insert(corr, gi, np.zeros(qj), axis=0)
    corr_uij = np.insert(corr_ui, gj, np.zeros(qi+1), axis=1)
    return corr_uij


# color compression

def compress_vector(v, p_cut = 0.0):
    compress_where = [i for i in range(len(v)) if v[i] <= p_cut]
    compress_ps = v[compress_where]
    compress_sum = np.sum(compress_ps)
    v_p = np.delete(v, compress_where)
    if len(compress_where):
        v_p = np.append(v_p, compress_sum)
    return v_p, compress_where, compress_ps


def uncompress_vector(v, compress_where, compress_ps, pseudo_count):
    if compress_where:
        v_0 = v[:-1]
        compressed_h = v[-1]

        if np.sum(compress_ps):
            delta_compressed_hs = np.log((compress_ps) / np.sum(compress_ps))
            compressed_hs = compressed_h + delta_compressed_hs
            compressed_hs[compress_ps==0] = np.log(pseudo_count) # unseen
        else:
            compressed_hs = np.ones(len(compress_ps)) * np.log(pseudo_count) # only unseen

        v_u = np.copy(v_0)
        for i in range(len(compress_where)):
            v_u = np.insert(v_u, compress_where[i], compressed_hs[i])
        return v_u
    else:
        return v


def compress_matrix(corr, where_i, where_j):
    qci = len(where_i)
    qcj = len(where_j)
    if qci == 0 and qcj == 0:
        return corr

    compressed_corr_i = corr[where_i, :]
    compressed_corr_j = corr[:, where_j]

    if qci:
        Q_corr_i = np.sum(compressed_corr_i, 0)
        Q_corr_i = np.delete(Q_corr_i, where_j)
    else:
        Q_corr_i = []

    if qcj:
        Q_corr_j = np.sum(compressed_corr_j, 1)
        Q_corr_j = np.delete(Q_corr_j, where_i)
    else:
        Q_corr_j = []

    if qci and qcj:
        Q_corr_ij = np.sum(compressed_corr_j[where_i])
    else:
        Q_corr_ij = []

    # delete rows and columns corresponding to compressed colors
    C = np.delete(corr, where_i, axis=0)
    C = np.delete(C, where_j, axis=1)

    # stack the compressed colors at the end of the matrix
    [qi, qj] = np.shape(C)
    if qci:
        C = np.insert(C, qi, Q_corr_i, axis=0)
        Q_corr_j = np.append(Q_corr_j, Q_corr_ij)
    if qcj:
        C = np.insert(C, qj, Q_corr_j, axis=1)

    return C


def uncompress_matrix(J, where_i, where_j):
    qci = len(where_i)
    qcj = len(where_j)

    if qci and qcj:
        J_clean = J[:-1, :-1]
        J_compressed_i = J[-1, :-1]  # without the compressed-compressed J_ij
        J_compressed_j = J[:-1, -1]
        J_compressed_ij = J[-1, -1]
    elif qci:
        J_clean = J[:-1, :]
        J_compressed_i = J[-1, :] # the whole line
        J_compressed_j = J[:-1, -1]
        J_compressed_ij = J[-1, -1]
    elif qcj:
        J_clean = J[:, :-1]
        J_compressed_i = J[-1, :-1]
        J_compressed_j = J[:, -1] # the whole line
        J_compressed_ij = J[-1, -1]
    else:
        return J

    # insert the compressed_i vector in the matrix
    J_u = np.copy(J_clean)
    for i in range(qci):
        J_u = np.insert(J_u, where_i[i], J_compressed_i, axis=0)
        # insert the compressed_ij in the vector j
        J_compressed_j = np.insert(J_compressed_j, where_i[i], J_compressed_ij)

    # insert the compressed_j vector in the matrix
    for j in range(qcj):
        J_u = np.insert(J_u, where_j[j], J_compressed_j, axis=1)

    return J_u

