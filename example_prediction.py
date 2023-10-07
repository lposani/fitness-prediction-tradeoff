from source import *
family = 'RNA-bind'

# read the MSA and wildtype
msa = read_MSA(family)
wildtype = read_WT(family)

# read mutation fitness from experimental data
mutated_sequence, real_deltafitness = read_mutations(family)

# infer an independent Potts model from the MSA
h, J = infer_independent_model(msa, W=np.ones(len(msa)))

# compute the predicted mutational fitnesses
inferred_deltafitness = np.zeros(len(mutated_sequence))

E_wt = Cpotts_energy(wildtype, h, J, msa.shape[1])
for i in range(len(mutated_sequence)):
    inferred_deltafitness[i] = E_wt - Cpotts_energy(mutated_sequence[i], h, J, msa.shape[1])

# visualize
f = plt.figure(figsize=(4,4))
plt.scatter(real_deltafitness, inferred_deltafitness, alpha=0.2, s=5)
plt.xlabel('Fitness (i,a) (experimental)')
plt.ylabel('$\Delta\mathcal{H}(i,a)$')
spear, p = spearmanr(real_deltafitness, inferred_deltafitness)
plt.title('%s   %s   $\\rho = %.4f$' % (family, 'IND', spear))
sns.despine(f)
plt.show()
