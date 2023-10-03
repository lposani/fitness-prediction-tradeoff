from source import *
import numpy as np

fm = ['o', 's', '^', '+', '*']


def focusing(family, model='ACEK', ds=None, K=0, differential_unseen=False, save=False, J0=0):
    distances = {
        'WW': np.arange(5, 30),
        'RNA-bind': np.arange(10, 81),
        'RL401': np.arange(10, 70),
        'BLAT': np.arange(10, 205),
        'PDZ': np.arange(10, 83),
        'DNA-bind': np.arange(10, 62),
        'UBOX': np.arange(10, 75)
    }
    if ds is None:
        ds = distances[family]
    msa = fasta.read_MSA(family)
    B, N = msa.shape
    wt = fasta.read_WT(family)
    mkdir('./data/%s/inference/focusing_performance_saved/' % family)
    spearmans = np.zeros(len(ds))
    Bs = np.zeros(len(ds))
    variances = np.zeros(len(ds))
    snrs = np.zeros(len(ds))
    mean_ds = np.zeros(len(ds))

    if differential_unseen:
        p_prior = np.zeros((N, 21))
        for i, d in enumerate(np.sort(ds)):
            W = hard_cutoff_weights(msa, wt, d)
            p = compute_weighted_averages(msa, W)
            print("Filling prior for %u unseen" % np.sum(p == 0))
            p_prior[p_prior == 0] = p[p_prior == 0]
    else:
        p_prior = []

    for i, d in enumerate(ds):
        W = hard_cutoff_weights(msa, wt, d)
        msa_cut = msa[W > 0]

        w_filename = './data/%s/inference/focusing_performance_saved/W_%u.dat' % (family, d)
        W = conditional_load(w_filename, lambda: compute_weights_MSA(msa_cut, params['reweighting']))

        spearman, variance, snr = focusing_performance(family, model, msa_cut, W, d, save=save, K=K, p_prior=p_prior,
                                                       du=differential_unseen)
        snrs[i] = snr
        Bs[i] = np.sum(W)
        variances[i] = variance
        spearmans[i] = spearman
        dists = np.sum(wt != msa_cut, 1)
        mean_ds[i] = np.average(dists, weights=W)

        print("[focusing]\t %s K=%u --- computing cutoff distance %.2f --- spear = %.2f, var = %.2f, SNR = %.1f" % (
            family, K, d, spearmans[i], variances[i], snrs[i]))

    if J0:
        visualize_focusing(family, model, K, ds=np.asarray(ds), spearmans=spearmans, Bs=Bs, variances=variances,
                           snrs=snrs, mean_ds=mean_ds, J0=J0)
    return ds, spearmans, Bs, variances, snrs, mean_ds


def focusing_performance(family, model, msa, W, d, save=False, K=0, plot=False, plot_ax='none', du=False, p_prior=[]):
    contact_type = params['contacts']
    rw = params['reweighting']
    unseen = params['unseen_pseudocount']
    overwrite = params['focusing_overwrite']
    variance_cutoff = params['variance_cutoff']

    savename = '../Data/%s/inference/focusing_performance_saved/%s_d=%.2f_rw=%.3f_K=%u_%s_du=%u_us=%.1f_vc=%.2f_snr.dat' \
               % (family, model, d, rw, K, contact_type, du, unseen, variance_cutoff)

    if os.path.exists(savename) and save == True and overwrite == False:
        data = np.loadtxt(savename)
        print('[focusing]\t --- loading data ---')
        return data[0], data[1], data[2]

    if 'IND' in model:
        h, J = inference.infer_independent_model(msa, W, p_prior=p_prior)
        model_title = 'IND_d=%.2f' % d
        variance = biasvariance.variance(msa, family, model, W=W)

    elif 'ACEK' in model:
        if contact_type == 'best':
            C = inference.best_contacts_family(family, K)
        else:
            C = inference.contacts_family(family, K)
        h, J = inference.infer_potts_ACEC(msa, C, W, p_prior=p_prior)
        model_title = 'ACEC_K=%u_d=%.2f' % (K, d)
        variance = biasvariance.variance(msa, family, model, contacts=C, W=W)

    else:
        print("invalid model")
        return -1

    s, realf, infef = landscape.compare_inferred_real(family, h, J, False)
    sig = np.var(infef.flatten())
    snr = np.sqrt(sig / variance)

    if plot:
        flag = 0
        if plot_ax == 'none':
            f, plot_ax = plt.subplots(figsize=(4, 4))
            flag = 1

        plot_ax.scatter(realf, infef, alpha=0.2)
        plot_ax.set_xlabel('$\Delta$fitness (experimental)')
        plot_ax.set_ylabel('$\Delta$fitness (inferred)')
        plot_ax.set_title('%s   %s   $\\rho = %.4f$' % (family, model_title, s))
        if flag:
            mkparent('../Plots/%s/focusing/plots/%s.pdf' % (family, model_title))
            plt.savefig('../Plots/%s/focusing/plots/%s.pdf' % (family, model_title))
            plt.close()

    if save:
        mkparent(savename)
        np.savetxt(savename, [s, variance, snr])

    return s, variance, snr


# visualization

def visualize_focusing(family, model, K, ds, spearmans, Bs, variances, snrs, mean_ds, J0, ax='none',
                       differential_unseen=False):
    mkdir('../plots/focusing')
    folder = './plots/focusing/rw=%.2f_vc=%.2f_us=%.2f_%s' % (params['reweighting'], params['variance_cutoff'],
                                                               params['unseen_pseudocount'], params['contacts'])
    mkdir(folder)
    if 'IND' in model:
        K = 0
    wt = fasta.read_WT(family)
    N = len(wt)
    msa = fasta.read_MSA(family)
    if 'IND' in model:
        mask = Bs > params['focusing_min_B_IND']
    elif 'ACEK' in model:
        mask = Bs > params['focusing_min_B_ACEK']
    allds = ds
    allsp = spearmans
    ds = ds[mask]
    spearmans = spearmans[mask]
    Bs = Bs[mask]
    variances = variances[mask]
    mean_ds = mean_ds[mask]
    snrs = snrs[mask]
    print(family, Bs)

    f, axs = plt.subplots(3, 1, figsize=(4, 5), gridspec_kw={'height_ratios': [1.7, 1, 1]}, sharex=True)
    axs[0].plot(ds, spearmans, linestyle='-', marker='.', color=pltcolors[0])
    axs[0].set_title('%s (%s, K=%u)' % (family, model, K))
    axs[0].set_ylabel(' $\\rho$ single mutations')

    p_prior = np.zeros((N, 21))
    if differential_unseen:
        for i, d in enumerate(ds):
            W = hard_cutoff_weights(msa, wt, d)
            p = inference.compute_weighted_averages(msa, W)
            p_prior[p_prior == 0] = p[p_prior == 0]

    bv_tradeoff = mean_ds * J0 + variances
    max_predicted = ds[np.argmin(bv_tradeoff)]
    max_real = ds[np.argmax(spearmans)]
    pred_performance_snr = spearmans[np.argmin(np.abs(snrs - params['focusing_snr_cutoff']))]
    max_snr = ds[np.argmin(np.abs(snrs - params['focusing_snr_cutoff']))]

    axs[0].axvline([max_real], color='k', linestyle='--')
    axs[0].axvline([max_predicted], color='r', linestyle='--')
    sns.despine(ax=axs[0])

    axs[1].set_ylabel('$\mu^2$ and $\sigma^2$')

    axs[1].plot(ds, variances, linestyle='-', marker='', color='k')
    axs[1].plot(ds, mean_ds * J0, linestyle='-', marker='', color='k')
    axs[1].plot(ds, bv_tradeoff, linestyle='-', marker='', color=pltcolors[3])
    axs[1].text(ds[0] - 0.2, variances[0], '$\\sigma^2$', ha='right')
    axs[1].text(ds[0] - 0.2, mean_ds[0] * J0, '$\mu^2 = J_0 D$', ha='right')
    axs[1].text(ds[0] - 0.2, bv_tradeoff[0], '$\mu^2 + \\sigma^2 $', ha='right')

    # axs[1].axvline([max_real], color='k', linestyle='--')
    axs[1].axvline([max_predicted], color='r', linestyle='--')
    sns.despine(ax=axs[1])
    # axs[1].legend(prop={'size': 8})

    fasta.histogram_distance(family, axs[2], reweighting=params['reweighting'])
    # axs[2].plot(ds, Bs, linestyle='-', color=pltcolors[1])
    axs[2].axvline([max_real], color='k', linestyle='--', label='Optimal cutoff $D^{opt}$')
    axs[2].axvline([max_predicted], color='r', linestyle='--', label='Predicted cutoff $D^{bv}$')
    axs[2].set_ylabel('$B(D_0)$')
    # axs[2].set_yscale('log')
    # axs[2].legend(prop={'size': 8})
    axs[2].set_xlabel('Cutoff distance $D_0$')
    axs[2].set_title('')
    sns.despine(ax=axs[2])

    f.savefig('%s/focusing_%s_%s_K=%u.pdf' % (folder, family, model, K))

    if ax != 'none':
        ax.plot(allds, allsp, linestyle='-', marker='.', markersize=3, color=pltcolors[0])
        ax.set_title('%s' % family, fontsize=10)
        ax.set_ylabel('$\\rho$ single mutations')
        ax.set_xlabel('$D_0$')
        ax.set_xlim([0, N + 1])
        # ax.axhline([marks[family]], color='grey', linestyle='--')
        ys = [0.35, 0.72]
        ax.plot([max_real, max_real], [spearmans[-1], np.max(spearmans)], color='k', linestyle='--', marker='_',
                label=' $\Delta\\rho^{opt} = %.3f$' % (np.max(spearmans) - spearmans[-1]))
        ax.plot([max_predicted, max_predicted], [spearmans[-1], spearmans[np.argmin(bv_tradeoff)]], color='b',
                linestyle='-', marker='_',
                label=' $\Delta\\rho^{bv} = %.3f$' % (spearmans[np.argmin(bv_tradeoff)] - spearmans[-1]))
        ax.plot([max_snr, max_snr], [spearmans[-1], pred_performance_snr], color='c', linestyle='-', marker='_',
                label=' $\Delta\\rho^{bv} = %.3f$' % (spearmans[np.argmin(bv_tradeoff)] - spearmans[-1]))

        # ax.legend(fontsize='x-small', loc='best')
        ax.set_ylim(ys)
        xs = ax.get_xlim()
        if spearmans[np.argmin(bv_tradeoff)] >= spearmans[-1]:
            ax.fill_between([xs[0], xs[1]], spearmans[-1], spearmans[np.argmin(bv_tradeoff)], color='g', alpha=0.15)
        else:
            ax.fill_between([xs[0], xs[1]], spearmans[-1], spearmans[np.argmin(bv_tradeoff)], color='r', alpha=0.15)
        ax.fill_between([xs[0], xs[1]], spearmans[np.argmin(bv_tradeoff)], np.max(spearmans), color='y', alpha=0.15)
        sns.despine()

    best_performance = np.max(spearmans)
    pred_performance = spearmans[np.argmin(bv_tradeoff)]
    print(len(spearmans), len(snrs))
    pred_performance_var = spearmans[np.argmin(np.abs(snrs - params['focusing_snr_cutoff']))]
    best_B = np.sum(hard_cutoff_weights(msa, wt, ds[np.argmax(spearmans)]))
    pred_B = np.sum(hard_cutoff_weights(msa, wt, ds[np.argmin(bv_tradeoff)]))

    from matplotlib import cm
    f = plt.figure(figsize=(3.5, 3))
    ax = f.add_subplot(111, projection='3d')
    ax.view_init(30, -145)
    zmin = np.min(bv_tradeoff) - 0.1
    ax.set_zlim([zmin, np.max(bv_tradeoff) + 0.2])
    ax.plot(mean_ds, variances, mean_ds * J0 + variances, linewidth=2, marker='', color=pltcolors[0])
    ax.plot(mean_ds, variances, mean_ds * J0 + variances, linewidth=2, marker='.', linestyle='', color='k')
    ax.plot([mean_ds[np.argmin(bv_tradeoff)]], [variances[np.argmin(bv_tradeoff)]],
            [J0 * mean_ds[np.argmin(bv_tradeoff)] + variances[np.argmin(bv_tradeoff)]], linestyle='', marker='.',
            color='r')
    ax.plot(mean_ds, variances, np.ones(len(variances)) * zmin, linewidth=2, marker='.', color='k', alpha=0.4)
    ax.plot([mean_ds[np.argmin(bv_tradeoff)]], [variances[np.argmin(bv_tradeoff)]], [zmin], linestyle='', marker='.',
            color='r', alpha=0.6)

    ax.plot([mean_ds[0], mean_ds[0]], [variances[0], variances[0]], [zmin, bv_tradeoff[0]], linewidth=1, linestyle='--',
            color='k', alpha=0.4)
    ax.plot([mean_ds[-1], mean_ds[-1]], [variances[-1], variances[-1]], [zmin, bv_tradeoff[-1]], linewidth=1,
            linestyle='--', color='k', alpha=0.4)
    ax.plot([mean_ds[np.argmin(bv_tradeoff)], mean_ds[np.argmin(bv_tradeoff)]],
            [variances[np.argmin(bv_tradeoff)], variances[np.argmin(bv_tradeoff)]], [zmin, np.min(bv_tradeoff)],
            linewidth=1, linestyle='--', color='r', alpha=0.6)

    ax.set_xlabel("$D$")
    ax.set_ylabel("$\sigma^2$")
    ax.set_zlabel("$J_0 D + \sigma^2$")
    ax.set_zlim([np.min(bv_tradeoff) - 0.1, np.max(bv_tradeoff) + 0.05])
    f.savefig('%s/focusing_%s_%s_K=%u_trajectory.pdf' % (folder, family, model, K))

    f = plt.figure(figsize=(3.5, 3))
    ax = f.add_subplot(111, projection='3d')
    ax.view_init(30, -145)
    zmin = np.min(spearmans) - 0.01
    ax.plot(mean_ds, variances, spearmans, linewidth=2, marker='', color=pltcolors[0])
    ax.plot(mean_ds, variances, spearmans, linewidth=2, marker='.', linestyle='', color='k')
    ax.plot([mean_ds[np.argmin(bv_tradeoff)]], [variances[np.argmin(bv_tradeoff)]],
            [spearmans[np.argmin(bv_tradeoff)]], linestyle='', marker='.',
            color='r')
    ax.plot(mean_ds, variances, np.ones(len(variances)) * zmin, linewidth=2, marker='.', color='k', alpha=0.4)
    ax.plot([mean_ds[np.argmin(bv_tradeoff)]], [variances[np.argmin(bv_tradeoff)]], [zmin], linestyle='', marker='.',
            color='r', alpha=0.6)

    ax.plot([mean_ds[0], mean_ds[0]], [variances[0], variances[0]], [zmin, spearmans[0]], linewidth=1, linestyle='--',
            color='k', alpha=0.4)
    ax.plot([mean_ds[-1], mean_ds[-1]], [variances[-1], variances[-1]], [zmin, spearmans[-1]], linewidth=1,
            linestyle='--', color='k', alpha=0.4)
    ax.plot([mean_ds[np.argmin(bv_tradeoff)], mean_ds[np.argmin(bv_tradeoff)]],
            [variances[np.argmin(bv_tradeoff)], variances[np.argmin(bv_tradeoff)]],
            [zmin, spearmans[np.argmin(bv_tradeoff)]],
            linewidth=1, linestyle='--', color='r', alpha=0.6)

    ax.set_xlabel("$D$")
    ax.set_ylabel("$\sigma^2$")
    ax.set_zlabel("$\\rho$")
    ax.set_zlim([zmin, np.max(spearmans) + 0.02])
    f.savefig('%s/focusing_%s_%s_K=%u_spearman.pdf' % (folder, family, model, K))

    def animate(i):
        ax.view_init(elev=10., azim=-3.14 / 4.)
        return f

    return best_performance, pred_performance, pred_performance_var, best_B, pred_B

