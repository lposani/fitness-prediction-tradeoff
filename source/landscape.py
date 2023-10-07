from .imports import *
from .fasta import *
from .inference import *
from .utilss import *
from .biasvariance import *


def compare_inferred_real(family, h, J, visualize=False, threshold_percentile=0):
    wildtype = read_WT(family)
    N = len(wildtype)
    mutated_sequence, real_deltafitness = read_mutations(family)
    if threshold_percentile:
        mask = real_deltafitness > np.percentile(real_deltafitness, threshold_percentile)
        mutated_sequence = mutated_sequence[mask]
        real_deltafitness = real_deltafitness[mask]

    inferred_deltafitness = np.zeros(len(mutated_sequence))
    E_wt = Cpotts_energy(wildtype, h, J, N)
    for i in range(len(mutated_sequence)):
        inferred_deltafitness[i] = E_wt - Cpotts_energy(mutated_sequence[i], h, J, N)
    spear, p = spearmanr(real_deltafitness, inferred_deltafitness)

    if visualize:
        visualize_inferred_real(real_deltafitness, inferred_deltafitness, family)

    return spear, real_deltafitness, inferred_deltafitness


def visualize_inferred_real(real_deltafitness, inferred_deltafitness, family, model_title=''):
    plt.figure(figsize=(5,5))
    plt.scatter(real_deltafitness, inferred_deltafitness, alpha=0.2)
    plt.xlabel('Fitness (i,a) (experimental)')
    plt.ylabel('$\Delta\mathcal{H}(i,a)$')
    spear, p = spearmanr(real_deltafitness, inferred_deltafitness)
    plt.title('%s   %s   $\\rho = %.4f$' % (family, model_title, spear))
    plt.tight_layout()
    plt.show()
