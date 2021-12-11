"""

test_models_bsd.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Thu 17 Dec 2020 10:40:05 EST

Description:

"""

import micro21cm
import numpy as np
import matplotlib.pyplot as pl

def test(rtol=1e-4):
    model = micro21cm.BubbleModel(bubbles_pdf='lognormal')

    R = 2.
    darr = np.array([0.5, 1., 2., 4])
    Varr = []

    for d in darr:
        Vo_gen = model.get_overlap_vol_generic(d=d, r1=R, r2=R)
        Vo = model.get_overlap_vol(d=d, R=R)
        Varr.append(Vo)

        if Vo == Vo_gen == 0:
            continue

        err = abs(Vo_gen - Vo) / Vo_gen

        assert err < rtol, "Vo_gen={} != Vo={}".format(Vo_gen, Vo)

    # Test different use cases
    Vo_1 = model.get_overlap_vol_generic(d=d, r1=R, r2=R*2)
    Vo_2 = model.get_overlap_vol_generic(d=d, r1=R*2, r2=R)
    assert Vo_1 == Vo_2

    Vo = model.get_overlap_vol_generic(d=darr, r1=R, r2=R)
    assert np.allclose(Vo, Varr)


if __name__ == '__main__':
    test()
