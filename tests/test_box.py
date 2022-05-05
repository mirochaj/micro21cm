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

def test(Lbox=50, use_h5py=True):

    # Initialize bubble model
    model = micro21cm.Box(use_h5py=use_h5py)

    # Check convenience routine that runs boxes over multiple Q
    model.generate_boxes(Q=np.array([0.1, 0.5]), z=z, Lbox=Lbox, Ts=Ts,
        R=kw['R'], sigma=kw['sigma'])

    # Make sure we find boxes that exist already.
    model.generate_boxes(Q=np.array([0.1, 0.5]), z=z, Lbox=Lbox, Ts=Ts,
        R=kw['R'], sigma=kw['sigma'])

    # Use individual box-generation routines. Should load ionization box.
    dTb = model.get_box_21cm(z=z, Lbox=Lbox, Ts=Ts, seed=1234,
        seed_rho=5678, **kw)
    dTb2 = model.get_box_21cm(z=z, Lbox=Lbox, Ts=Ts, seed=1234,
        seed_rho=5678, **kw)

    assert np.allclose(dTb, dTb2)

    box, box_tot = model.get_box_bubbles(z=z, Lbox=Lbox, **kw)
    box2, box_tot2 = model.get_box_bubbles(z=z, Lbox=Lbox, **kw)

    assert np.allclose(box, box2)

    # Check that the bubble filling factor is consistent.
    Q_box = 1. - box.sum() / float(box.size)
    assert abs(Q_box - kw['Q']) < 0.05

    # Test smoothing routines, check variance.
    box_sm1 = micro21cm.util.smooth_box(box, R=1, periodic=True).real
    var1 = np.std(box_sm1.ravel())**2

    box_sm2 = micro21cm.util.smooth_box(box, R=2, periodic=True).real
    var2 = np.std(box_sm2.ravel())**2

    assert var2 < var1

    # Get random box
    rbox1 = model.get_box_rand(box=box, seed=1234, Lbox=Lbox, Q=kw['Q'])

    # Default vox=1 so this should be equal to rbox1
    rbox2 = model.get_box_rand(seed=1234, Lbox=Lbox, Q=kw['Q'])

    assert np.all(rbox1 == rbox2)

    # This should raise an error, unlikely we get the right Q with so few pixels
    try:
        rbox3 = model.get_box_rand(seed=1234, Lbox=5, vox=1, Q=kw['Q'])
    except ValueError:
        assert True

if __name__ == '__main__':
    test(use_h5py=True)
    test(use_h5py=False)
