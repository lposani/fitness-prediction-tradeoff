from .utilss import *
from .fasta import *
from .inference import *
from .landscape import compare_inferred_real


def variance(msa, family, model, W=None, contacts=[], return_all=False):
    mutated = is_mutated(family)
    wt = read_WT(family)
    N, A = mutated.shape

    if W is None:
        W = np.ones(len(msa))
    Beff = np.sum(W)

    threshold = params['variance_cutoff'] / float(len(W))
    if 'IND' in model or len(contacts) == 0:
        p = compute_weighted_averages(msa, W)
        p[p < threshold] = 0
        p[p == 0] = params['unseen_pseudocount'] * np.min(p[p > 0])

        pwt = np.asarray([p[i, wt[i]] for i in range(N) if p[i, wt[i]] > 0])

        p_nwt = np.asarray([p[i, a] for i in range(N) for a in range(A) if
                            (a != wt[i]) and (p[i, a]) > 0 and (mutated[i, a] > 0)])

        theor_variance = 1. / Beff * (np.nanmean((1. / pwt) + np.nanmean(1. / p_nwt)))
        if return_all:
            mutated_sequence, real_deltafitness = read_mutations(family)
            theor_variances = []
            for seq in mutated_sequence:
                for i in range(N):
                    if seq[i] != wt[i]:
                        theor_variances.append(1. / Beff * (1. / p[i, seq[i]] + 1. / p[i, wt[i]]))
            return theor_variances

    elif 'ACEK' in model:
        C = contacts_to_matrix(contacts, N)
        p, corr, Z = compute_weighted_moments_contacts(msa, W, C)
        p[p < threshold] = 0
        corr[corr < threshold] = 0
        p[p == 0] = params['unseen_pseudocount'] * np.min(p[p > 0])
        corr[corr == 0] = params['unseen_pseudocount'] * np.min(corr[corr > 0])

        v = np.zeros((N, A))
        variances = []

        for i in range(N):
            nc = np.sum(C[i, :])
            for a in range(A):
                if p[i, a] > 0 and p[i, wt[i]] > 0 and mutated[i, a] > 0:
                    v[i, a] = max((nc - 1), 1) * (1. / p[i, a] + 1. / p[i, wt[i]])
                    for j in range(N):
                        if C[i, j] and corr[i, j, a, wt[j]] and corr[i, j, wt[i], wt[j]]:
                            v[i, a] += 1. / corr[i, j, a, wt[j]] + 1. / corr[i, j, wt[i], wt[j]]
                    variances.append(v[i, a] / Beff)

        theor_variance = np.nanmean(variances)

    return theor_variance


def subsample_msa_window(family, distance, B, msa='none'):
    wt = read_WT(family)
    N = float(len(wt))
    if msa == 'none':
        msa = read_MSA(family)

    dists = np.sum(wt != msa, 1)
    order = np.argsort(dists)
    dists_ordered = dists[order]
    dists_sum = np.cumsum(dists_ordered)
    for i in range(B, len(msa)):
        d_try = (dists_sum[B + i] - dists_sum[i]) / float(B)
        if d_try >= distance:
            msa_try = msa[order][i:i + B]
            break
    return msa_try, d_try


def scaling_descriptors(family, model, B, dist, niter, contacts=None, mdm_flag=False, main_distance_matrix=[]):

    print("[scaling]\t O> computing %s \t \t \t B=%u  D = %.2f" % (family, B, dist))
    spearmans = np.zeros(niter)
    distances = np.zeros(niter)
    variances = np.zeros(niter)
    msa_Bs = np.zeros(niter)
    if contacts is None:
        contacts = []

    for n in range(niter):
        t0 = time.time()
        if mdm_flag:
            sub_msa, D_eff, W = subsample_msa(family, dist, int(B), reweighting=params['reweighting'],
                                              main_distance_matrix=main_distance_matrix)
        else:
            sub_msa, D_eff, W = subsample_msa(family, dist, int(B), reweighting=params['reweighting'])
        if params['timing']:
            print("[timing]\t sampling (%s): %.4f" % (family, time.time() - t0))

        if abs(D_eff - dist) < 0.05:
            t0 = time.time()
            if 'IND' in model:
                h, J = infer_independent_model(sub_msa, W)
                if params['timing']:
                    print("[timing]\t inference (%s): %.4f" % (family, time.time() - t0))
                t0 = time.time()
                variances[n] = variance(sub_msa, family, model, W=W)
            if 'ACEK' in model:
                h, J = infer_potts_ACEC(sub_msa, contacts, W)
                if params['timing']:
                    print("[timing]\t inference (%s): %.4f" % (family, time.time() - t0))
                t0 = time.time()
                variances[n] = variance(sub_msa, family, model, contacts=contacts, W=W)
            t0 = time.time()
            spearmans[n], dfr, dfi = compare_inferred_real(family, h, J)
            if params['timing']:
                print("[timing]\t spearman (%s): %.4f" % (family, time.time() - t0))
            distances[n] = D_eff
            msa_Bs[n] = np.sum(W)
        else:
            variances[n] = np.nan
            spearmans[n] = np.nan
            distances[n] = np.nan
            msa_Bs[n] = np.nan

    print("[scaling]\t O> computed %s \t \t \t B=%u  Beff = %.1f  D = %.2f Deff=%.2f  var = %.2f  spearm = %.3f" % (
        family, B, np.mean(msa_Bs), dist, np.mean(distances), np.mean(variances), np.mean(spearmans)))

    return spearmans, distances, variances, msa_Bs


def scaling_bias_variance(family, Ds, Bs, model='ACEK', K=0, plot=True, plot_ax='auto', niter=0):
    if niter == 0:
        niter = params['scaling_niter']
    spearmans = np.zeros((len(Bs), len(Ds), niter))
    distances = np.zeros((len(Bs), len(Ds), niter))
    variances = np.zeros((len(Bs), len(Ds), niter))
    msa_Bs = np.zeros((len(Bs), len(Ds), niter))
    contacts_type = params['contacts']

    B, N = B_N(family)

    folder = f'./data/{family}/subsampling/'
    mkdir(folder)

    # if 'ACEK' in model and K == 0:
    #     model = 'IND'

    if 'ACEK' in model:
        if contacts_type == 'F':
            contacts = contacts_family(family, K)
        if contacts_type == 'best':
            contacts = best_contacts_family(family, K)
    else:
        contacts_type = 'x'
        contacts = []

    cache_name = './data/%s/subsampling/%s_n=%u_us=%.1f_ct=%s_K=%u_rw=%.2f_vt=%.2f_Bs=%u-%u.dat' % (
        family, model, niter, params['unseen_pseudocount'], contacts_type, K, params['reweighting'],
        params['variance_cutoff'], Bs[0], Bs[-1])
    print(cache_name)
    if os.path.exists(cache_name):
        msa_Bs, spearmans, variances, distances = pickle.load(open(cache_name, 'rb'))
    else:
        msa = read_MSA(family)
        (B, N) = msa.shape
        if B < 20000 and plot:
            def main_distance_matrix(msa):
                (B, N) = np.shape(msa)
                D = np.zeros((B, B))
                for i, s in enumerate(msa):
                    D[i] = np.sum(s != msa, 1) / float(N)
                return D

            main_distance_matrix = main_distance_matrix(msa)
            mdm_flag = True
        else:
            mdm_flag = False
            main_distance_matrix = []

        for i, B in enumerate(Bs):
            for j, dist in enumerate(Ds):
                spearmans[i, j, :], distances[i, j, :], variances[i, j, :], msa_Bs[i, j, :] = scaling_descriptors(
                    family=family,
                    model=model, B=B,
                    dist=dist, niter=niter,
                    contacts=contacts,
                    mdm_flag=mdm_flag,
                    main_distance_matrix=main_distance_matrix, )

        pickle.dump([msa_Bs, spearmans, variances, distances], open(cache_name, 'wb'))

    mask = (np.nanmean(variances, 2).flatten() < params['max_variance_scaling'][model])
    mean_spearmans = np.nanmean(spearmans, 2).flatten()[mask]
    mean_distances = np.nanmean(distances, 2).flatten()[mask]
    mean_variances = np.nanmean(variances, 2).flatten()[mask]
    mean_Bs = np.nanmean(msa_Bs, 2).flatten()[mask]

    if plot:
        J0, J0_std = visualize_scaling(mean_spearmans, mean_variances, mean_distances, mean_Bs, family=family,
                                       model=model, K=K, cax=plot_ax, min_B=np.min(Bs))
    else:
        J0, J0_std = find_best_J0(mean_spearmans, mean_variances, N * mean_distances)

    print(J0, J0_std)

    np.savetxt(cache_name[:-4] + '_J0.dat', [J0, J0_std[0], J0_std[1]])

    return mean_Bs, mean_spearmans, mean_variances, mean_distances, J0, J0_std


def visualize_scaling(spearmans, variances, distances, Bs, family, model, K=0, min_B=0, bins=16, cax='auto'):
    spearmans = spearmans.flatten()
    variances = variances.flatten()
    distances = distances.flatten()
    Bs = Bs.flatten()

    folder = './plots/scaling/rw=%.2f_vc=%.2f_us=%.2f_%s' % (params['reweighting'], params['variance_cutoff'],
                                                             params['unseen_pseudocount'], params['contacts'])
    mkdir(folder)

    wt = read_WT(family)
    N = len(wt)

    # msa_D, msa_var, msa_spearman = conditional_load('../Data/%s/inference/standard_msa_%s_K=%u_rw=%.2f_us=%.2f.dat' %
    #                                                 (family, model, K, reweighting, unseen),
    #                                                 lambda: standard_msa_bias_variance(family, model, K, reweighting=reweighting, unseen=unseen))
    msa_D, msa_var, msa_spearman = standard_msa_bias_variance(family, model, K)
    spearmans = spearmans
    distances = distances * N
    variances = variances

    # binned stuff
    mean_spearmans, bin_variances, bin_distances, nn = scipy.stats.binned_statistic_2d(variances, distances, spearmans,
                                                                                       bins=bins,
                                                                                       statistic=lambda x: np.nanmean(
                                                                                           x))
    err_spearmans, bin_variances, bin_distances, nn = scipy.stats.binned_statistic_2d(variances, distances, spearmans,
                                                                                      bins=bins,
                                                                                      statistic=lambda x: np.nanstd(x))

    bin_variances = bin_variances[:-1] + (bin_variances[1] - bin_variances[0]) / 2.
    bin_distances = bin_distances[:-1] + (bin_distances[1] - bin_distances[0]) / 2.
    X, Y = np.meshgrid(bin_distances, bin_variances)

    fb_spearmans = mean_spearmans[np.isnan(mean_spearmans) == 0]
    fb_variances = Y[np.isnan(mean_spearmans) == 0]
    fb_distances = X[np.isnan(mean_spearmans) == 0]
    fb_spearmans_err = err_spearmans[np.isnan(mean_spearmans) == 0]

    best_fit, best_fit_std = find_best_J0(spearmans, variances, distances)
    if params['debug']:
        print("%s - non-binned J0 = %.4f" % (family, best_fit))

    best_fit_bin, best_fit_std_bin = find_best_J0(fb_spearmans, fb_variances, fb_distances)
    if params['debug']:
        print("%s binned J0 = %.4f" % (family, best_fit_bin))

    f = plt.figure(figsize=(16, 4))

    plt.subplot(141)
    spear, p = scipy.stats.spearmanr(best_fit_bin * fb_distances + fb_variances, fb_spearmans)
    plt.errorbar(best_fit_bin * fb_distances + fb_variances, fb_spearmans, fb_spearmans_err, alpha=0.8, linestyle='',
                 marker='o')
    [slope, intercept] = np.polyfit(best_fit_bin * fb_distances + fb_variances, fb_spearmans, 1)
    x = np.asarray(plt.gca().get_xlim())
    # plt.plot(x, slope * x + intercept, 'r')
    plt.title('%s (%s) $r_s = %.2f$' % (family, model, spear))
    plt.xlabel('$J_0$ $D$ + $\sigma^2(B)$')
    plt.ylabel('$\\rho$ single mutations (binned)')
    # plt.ylim([0.3, 0.65])
    # plt.xlim([min(xx) * 0.95, max(xx) * 1.05])
    plt.ylim([min(spearmans) - 0.05, max(spearmans) + 0.05])

    plt.subplot(142)
    spear, p = scipy.stats.spearmanr(best_fit * distances + variances, spearmans)
    plt.scatter(best_fit * distances + variances, spearmans, alpha=0.5)
    plt.plot(best_fit * msa_D + msa_var, msa_spearman, alpha=1.0, color='r', marker='o', linestyle='')
    plt.title('%s (%s) $r_s = %.2f$, $n=%u$' % (family, model, spear, len(distances)))
    plt.xlabel('$J_0$ $D$ + $\sigma^2(B)$')
    plt.ylabel('$\\rho$ single mutations')
    plt.ylim([min(spearmans) - 0.05, max(spearmans) + 0.05])

    spear, p = scipy.stats.spearmanr(distances, spearmans)
    plt.subplot(143)
    plt.scatter(distances, spearmans, alpha=0.5)
    plt.plot(msa_D, msa_spearman, alpha=1.0, color='r', marker='o', linestyle='')
    plt.xlabel('$D$')
    plt.title('only D, $r_s = %.2f$' % spear)
    spear, p = scipy.stats.spearmanr(variances, spearmans)
    # plt.ylim([0.3, 0.65])

    plt.subplot(144)
    plt.scatter(variances, spearmans, alpha=0.5, label='sub MSA')
    plt.plot(msa_var, msa_spearman, alpha=1.0, color='r', label='full MSA', marker='o', linestyle='')
    plt.legend()
    plt.title('only $\sigma^2(B)$, r_s = %.2f' % spear)
    plt.xlabel('$\sigma^2(B)$')
    # plt.ylim([0.3, 0.65])

    # plt.tight_layout()
    plt.savefig('%s/Scaling_%s_%s_K=%u.pdf' % (folder, model, family, K))

    sns.despine(f)
    if cax != 'auto':
        spear, p = scipy.stats.spearmanr(best_fit_bin * fb_distances + fb_variances, fb_spearmans)
        # cax.errorbar(best_fit_bin * fb_distances + fb_variances, fb_spearmans, fb_spearmans_err, alpha=0.8, linestyle='',
        #             marker='o')
        cax.scatter(best_fit * distances + variances, spearmans, alpha=0.5, marker='.')
        spear, p = scipy.stats.spearmanr(best_fit * distances + variances, spearmans)
        cax.set_title('K=%u, $r_S=%.2f$' % (K, np.abs(spear)))
        cax.set_xlabel('$J_0$ $D$ + $\sigma^2$')
        cax.set_ylabel('$\\rho$ single mutations')
        # cax.set_ylim([min(spearmans) - 0.02, max(spearmans) + 0.02])
        xs = cax.get_xlim()
        ys = cax.get_ylim()
        cax.text(xs[0] + (xs[1] - xs[0]) * 0.6, ys[0] + (ys[1] - ys[0]) * 0.93, '$J_0$ = %.3f' % best_fit)

    return best_fit, best_fit_std  # , best_distance_bin, best_distance_bin_std


def visualize_bias_variance(spearmans, variances, distances_in, Bs, family, model, bins=8, min_B=1):
    spearmans = spearmans.flatten()
    variances = variances.flatten()
    distances_in = distances_in.flatten()
    Bs = Bs.flatten()

    Bs_plot = np.logspace(np.log10(min_B), np.log10(np.nanmax(Bs)), bins)
    N = len(read_WT(family))
    distances = N * distances_in
    ds_plot = np.linspace(np.nanmin(distances), np.nanmax(distances), bins)
    f, axs = plt.subplots(1, 2, figsize=(7, 3.5), sharey=True)
    delta_d = min([0.5 * (ds_plot[1] - ds_plot[0]), 2])
    delta_B = 0.2

    folder = '../Plots/descriptors/rw=%.2f_vc=%.2f_us=%.2f_%s' % (params['reweighting'], params['variance_cutoff'],
                                                                  params['unseen_pseudocount'], params['contacts'])
    mkdir(folder)

    for b in Bs_plot:
        means = []
        err = []
        for d in ds_plot:
            mask = (distances > d - delta_d) & (distances < d + delta_d) & (Bs < b + b * delta_B) & (
                    Bs > b - b * delta_B)
            means.append(np.nanmean(spearmans[mask]))
            err.append(np.nanstd(spearmans[mask]))
        R, p = spearNaN(distances[(Bs < b + b * delta_B) & (Bs > b - b * delta_B)],
                        spearmans[(Bs < b + b * delta_B) & (Bs > b - b * delta_B)])
        if params['debug']:
            print("Stats for D plot, B = %u:  spearman R = %.2f, p = %.1e" % (int(b), R, p))
        axs[0].errorbar(ds_plot, means, err, label='$B = %u$' % b, marker='o', linestyle='--')

    axs[0].legend()
    axs[0].set_xlabel('Mean Hamming distance $D$')
    axs[0].set_ylabel('Predictive performance $\\rho$')
    # axs[0].set_xticks([35, 40, 45, 50, 55, 60, 65])
    # axs[0].set_ylim([0.2, 0.60])
    # axs[0].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6])
    sns.despine(ax=axs[0])

    for d in ds_plot:
        means = []
        err = []
        for b in Bs_plot:
            mask = (distances > d - delta_d) & (distances < d + delta_d) & (Bs < b + b * delta_B) & (
                    Bs > b - b * delta_B)
            means.append(np.nanmean(spearmans[mask]))
            err.append(np.nanstd(spearmans[mask]))
        R, p = spearNaN(Bs[(distances < d + delta_d) & (distances > d - delta_d)],
                        spearmans[(distances < d + delta_d) & (distances > d - delta_d)])
        if params['debug']:
            print("Stats for B plot, d = %u:  spearman R = %.2f, p = %.1e" % (d, R, p))
        axs[1].set_xscale('log')
        axs[1].errorbar(Bs_plot, means, err, label='$D = %u$' % d, marker='o', linestyle='--')

    axs[1].legend()
    axs[1].set_xlabel('Number of sequences $B$')
    sns.despine(ax=axs[1])
    plt.savefig('%s/Descriptors_%s_%s.pdf' % (folder, model, family))


def standard_msa_bias_variance(family, model, K=0):
    msa = read_MSA(family)
    wt = read_WT(family)
    W = compute_reweighting(family, params['reweighting'])
    D = np.average(np.sum(wt != msa, 1), weights=W)
    Beff = np.sum(W)
    contacts_type = params['contacts']

    if 'ACEK' in model and K == 0:
        model = 'IND'
    if 'ACEK' in model:
        if contacts_type == 'F':
            contacts = contacts_family(family, K)
        if contacts_type == 'best':
            contacts = best_contacts_family(family, K)
    else:
        contacts_type = ''
    if 'IND' in model:
        h, J = infer_independent_model(msa, W)
        var = variance(msa, family, model, W=W)
    if 'ACEK' in model:
        t0 = time.time()
        h, J = infer_potts_ACEC(msa, contacts, W)
        if params['debug']:
            print("inference: %.2f" % (time.time() - t0))
        var = variance(msa, family, model, W=W, contacts=contacts)

    spearman, dfr, dfi = compare_inferred_real(family, h, J)
    return D, var, spearman


def Bs_family(family, model='ACEK'):
    msa = read_MSA(family)
    (B, N) = msa.shape
    if 'IND' in model:
        Bs = np.hstack([[20, 30, 40, 50], np.asarray((1. / np.linspace(1. / 60, 1. / (8000), 8)), dtype=int)[:-1],
                        [800, 1200, 1600]])
    elif 'ACEK' in model:
        Bs = np.hstack(
            [[20, 40, 80, 160], np.asarray((1. / np.linspace(1. / 300, 1. / (10000), 8)), dtype=int)[:-1]])

    Bs = Bs[Bs < B]
    Ds = np.arange(params['dmin'], params['dmax'], 0.025)
    return Bs, Ds


def find_best_J0(spearmans, variances, distances, interval=None, finesse=10000):
    if interval is None:
        interval = [0, 1]
    alphas = np.linspace(interval[0], interval[1], finesse)
    ss = np.zeros(finesse)
    for i in range(finesse):
        # cc = np.corrcoef(alphas[i]*distances + variances, spearmans)
        # ss[i] = cc[0,1]
        cc = scipy.stats.spearmanr(alphas[i] * distances + variances, spearmans)
        ss[i] = cc[0]
        # print alphas[i], ss[i]

    best_fit = np.mean(alphas[ss == np.min(ss)])
    interval_alphas = alphas[np.abs(ss) > 0.99 * np.abs(np.min(ss))]
    print(interval_alphas)

    if best_fit > 0.9:
        interval = [0, 10]
        alphas = np.linspace(interval[0], interval[1], finesse)
        ss = np.zeros(finesse)
        for i in range(finesse):
            # cc = np.corrcoef(alphas[i]*distances + variances, spearmans)
            # ss[i] = cc[0,1]
            cc = scipy.stats.spearmanr(alphas[i] * distances + variances, spearmans)
            ss[i] = cc[0]
            # print alphas[i], ss[i]

        best_fit = np.mean(alphas[ss == np.min(ss)])
        interval_alphas = alphas[np.abs(ss) > 0.99 * np.abs(np.min(ss))]
    plt.figure()
    plt.plot(alphas, ss)
    plt.xlabel('J0')
    plt.ylabel('Spearman Corr')

    print(best_fit, [np.nanmin(interval_alphas), np.nanmax(interval_alphas)])

    return best_fit, [np.nanmin(interval_alphas), np.nanmax(interval_alphas)]
