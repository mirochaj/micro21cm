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

def test():
    model_logn = micro21cm.BubbleModel(bubbles_pdf='lognormal')
    model_plex = micro21cm.BubbleModel(bubbles_pdf='plexp')

    bsd_logn = model_logn.get_bsd(Q=0.1, R=2., sigma=0.5)
    bsd_plex = model_plex.get_bsd(Q=0.1, R=2., gamma=-3.5)

    fig, ax = pl.subplots(1, 1)

    ax.loglog(model_logn.tab_R, bsd_logn * model_logn.tab_R**4)
    ax.loglog(model_plex.tab_R, bsd_plex * model_plex.tab_R**4)
    ax.set_ylim(1e-8, 1e-1)

    pl.savefig('{!s}.png'.format(__file__[0:__file__.rfind('.')]))
    pl.close()


if __name__ == '__main__':
    test()
