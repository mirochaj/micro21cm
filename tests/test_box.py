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

    # Check convenience routine that runs boxes over multiple Q
    model.generate_boxes(Q=np.array([0.1, 0.5]), z=z, Lbox=Lbox, Ts=Ts,
        R=kw['R'], sigma=kw['sigma'])

    # Use individual box-generation routines. Should load ionization box.
    dTb = model.get_box_21cm(z=z, Lbox=Lbox, Ts=Ts, **kw)
    box, box_tot = model.get_box_bubbles(z=z, Lbox=Lbox, **kw)

    # Check that the bubble filling factor is consistent.
    Q_box = 1. - box.sum() / float(box.size)

    assert abs(Q_box - kw['Q']) < 0.05

    # Test smoothing routines, check variance.
    box_sm1 = micro21cm.util.smooth_box(box, R=1, periodic=True).real
    var1 = np.std(box_sm1.ravel())**2

    box_sm2 = micro21cm.util.smooth_box(box, R=2, periodic=True).real
    var2 = np.std(box_sm2.ravel())**2

    # Get random box
    rbox = model.get_box_rand(box, Lbox=Lbox, Q=kw['Q'])

    assert var2 < var1


if __name__ == '__main__':
    test()
