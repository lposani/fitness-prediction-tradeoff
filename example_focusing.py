from source import focusing
import matplotlib.pyplot as plt

family = 'RNA-bind'

# Using pre-computed J0 to compute the predicted cutoff.
J0 = 0.076

# Focusing results are loaded from a cached run, use save=False to recompute them from scratch
ds, spearmans, Bs, variances, snrs, mean_ds = focusing(family=family, model='IND', K=0, save=True, J0=J0)
plt.show()
