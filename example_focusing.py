from source import focusing
import matplotlib.pyplot as plt

family = 'RNA-bind'

# choose signal-to-noise ratio threshold to predict the optimal focusing
snr_threshold = 3.0

# Focusing results are loaded from a cached run, use save=False to recompute them from scratch
ds, spearmans, Bs, variances, snrs, mean_ds = focusing(family=family, model='IND',
                                                       save=True, snr_threshold=snr_threshold)
plt.show()
