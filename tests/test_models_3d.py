"""

test_models_3d.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Mon 21 Dec 2020 14:11:57 EST

Description:

"""

import micro21cm
import numpy as np
import powerbox as pbox
import matplotlib.pyplot as pl

# Model parameters to explore
redshift = 8.
Ts = 3.
bubble_kw = {'Q': 0.1, 'R_b': 2.5, 'sigma_b': 0.3}

# Initialize bubble model
model = micro21cm.BubbleModel(bubbles_ion=True)

# Set modes of interest
k_mic = np.logspace(-1., 1, 21)

# Compute P(k)
p_mic = model.get_ps_21cm(z=redshift, k=k_mic, Ts=Ts, **bubble_kw)

# Generate a realization of this model in a 3-d box.
Lbox = 100.
box, rho, dTb = model.get_3d_realization(z=redshift, Lbox=Lbox, Ts=Ts,
    **bubble_kw)

# Check that the bubble filling factor is consistent.
print("Q box: {:.3f}".format(1. - box.sum() / float(box.size)))
print("Q mod: {:.3f}".format(bubble_kw['Q']))

pmid = box.shape[0] // 2

# Plot the 21-cm field [thin 20 Mpc projection]
fig1, ax1 = pl.subplots(1, 1, num=1)
ax1.imshow(box[pmid-10:pmid+11].mean(axis=0).T, origin='lower',
    extent=(0, Lbox, 0, Lbox))

# Compare the power spectra computed from the bubble model to the one obtained
# from the 3-d realization.
fig2, ax2 = pl.subplots(2, 1, num=2)

ax2[0].loglog(k_mic, k_mic**3 * p_mic / 2. / np.pi**2, ls='-', color='k',
    label='~analytic')

# Get power spectrum of the box we made
p_box, k_box = pbox.get_power(dTb, Lbox, ignore_zero_mode=True)

ax2[0].loglog(k_box, k_box**3 * p_box / 2. / np.pi**2, ls='--', color='b',
    label='from box')

# Compute the ratio of the solutions. Interpolate to common k grid first.
ratio = p_mic / np.interp(k_mic, k_box, p_box)
ax2[1].semilogx(k_mic, ratio, color='k')
ax2[1].plot([5e-2, 3], [1]*2, color='k', ls=':')

# Add some labels etc.
ax2[0].legend(loc='upper left')
ax2[0].set_xlim(5e-2, 3)
ax2[1].set_xlim(5e-2, 3)
ax2[1].set_xlabel(micro21cm.labels['k'])
ax2[0].set_ylabel(micro21cm.labels['delta_sq_long'])
ax2[1].set_ylabel('ratio')
ax2[1].set_ylim(min(0.7, 0.9*min(ratio)), max(1.1*max(ratio), 1.3))
