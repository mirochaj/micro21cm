"""

test_models_3d.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Mon 21 Dec 2020 14:11:57 EST

Description:

"""

import micro21cm
import numpy as np

# Model parameters to explore
z = 8.
Ts = 3.
kw = {'Q': 0.1, 'R': 2.5, 'sigma': 1}

def test(Lbox=50):

    # Initialize bubble model
    model = micro21cm.Box()

    # Generate a realization of this model in a 3-d box.
    dTb = model.get_box_21cm(z=z, Lbox=Lbox, Ts=Ts, **kw)
    box, box_tot = model.get_box_bubbles(z=z, Lbox=Lbox, **kw)

    # Check that the bubble filling factor is consistent.
    Q_box = 1. - box.sum() / float(box.size)

    assert abs(Q_box - kw['Q']) < 0.05


if __name__ == '__main__':
    test()
