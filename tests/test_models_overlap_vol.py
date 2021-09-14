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
    for d in [0.5, 1., 2., 4.]:
        Vo_gen = model.get_overlap_vol_generic(d=d, r1=R, r2=R)
        Vo = model.get_overlap_vol(d=d, R=R)

        if Vo == Vo_gen == 0:
            continue

        err = abs(Vo_gen - Vo) / Vo_gen

        assert err < rtol, "Vo_gen={} != Vo={}".format(Vo_gen, Vo)

if __name__ == '__main__':
    test()
