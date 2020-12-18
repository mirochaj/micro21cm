"""

test.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Wed 16 Dec 2020 16:16:41 EST

Description:

"""

import micro21cm
import numpy as np
import matplotlib.pyplot as pl

# Setup 2-panel plot (one panel / band)
fig, ax = pl.subplots(1, 1)
fig.subplots_adjust(wspace=0.3)

k = np.logspace(-2, 1, 100)

igm = micro21cm.BubbleModel()

# Plot the limits first
for z in [6, 10, 15]:

    # This is the power + 2-sigma for each k
    #ax = plot_limits(band, ax=axes[band], color='k', fmt='o')

    # Compute the matter power spectrum at relevant redshift
    P_mm = igm.get_ps_matter(z, k)
    D_sq_mm = k**3 * P_mm / 2. / np.pi**2

    ax.loglog(k, D_sq_mm, label=r'$z={}$'.format(z))

ax.set_xlabel(micro21cm.labels['k'])
ax.set_ylabel(micro21cm.labels['delta_sq'])
ax.set_ylim(1e-5, 1e1)
ax.legend()