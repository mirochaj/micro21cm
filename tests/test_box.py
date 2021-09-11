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

# Model parameters to explore
z = 8.
Ts = 3.
kw = {'Q': 0.1, 'R': 2.5, 'sigma': 1}

def test(Lbox=100):

    # Initialize bubble model
    model = micro21cm.Box()

    # Generate a realization of this model in a 3-d box.
    dTb = model.get_box_21cm(z=z, Lbox=Lbox, Ts=Ts, **kw)
    box, box_tot = model.get_box_bubbles(z=z, Lbox=Lbox, **kw)

    # Check that the bubble filling factor is consistent.
    print("Q box: {:.3f}".format(1. - box.sum() / float(box.size)))
    print("Q mod: {:.3f}".format(kw['Q']))

    pmid = box.shape[0] // 2

    # Compare the power spectra computed from the bubble model to the one obtained
    # from the 3-d realization.
    #p_box, k_box = pbox.get_power(dTb, Lbox, ignore_zero_mode=True)

    ## Compute the ratio of the solutions. Interpolate to common k grid first.
    #ratio = p_mic / np.interp(k_mic, k_box, p_box)

if __name__ == '__main__':
    test()
