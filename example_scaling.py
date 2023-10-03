import matplotlib.pyplot as plt
from source import Bs_family, B_N, scaling_bias_variance

family = 'RNA-bind'

# Results are loaded from a cached run, delete or rename the
#   folder data/{family}/subsampling to rerun from scratch

# Independent model
Bs, Ds = Bs_family(family, model='IND')
res = scaling_bias_variance(family=family, model='IND', Ds=Ds, Bs=Bs, K=0, plot=True)

plt.show()

# Sparse Potts model
Bs, Ds = Bs_family(family, model='ACEK')
B, N = B_N(family)
res = scaling_bias_variance(family=family, model='ACEK', Ds=Ds, Bs=Bs, K=N, plot=True)

plt.show()
