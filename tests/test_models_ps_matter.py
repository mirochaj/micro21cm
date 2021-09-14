"""

test.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Wed 16 Dec 2020 16:16:41 EST

Description:

"""

import micro21cm
import numpy as np


def test():
    k = np.logspace(-2, 1, 100)

    igm = micro21cm.BubbleModel()

    # Plot the limits first
    P_mm = np.inf
    for z in [6, 10, 15]:

        # This is the power + 2-sigma for each k
        #ax = plot_limits(band, ax=axes[band], color='k', fmt='o')

        # Compute the matter power spectrum at relevant redshift
        P_mm_z = igm.get_ps_matter(z, k)

        assert np.all(P_mm_z < P_mm)
        P_mm = P_mm_z


if __name__ == '__main__':
    test()
