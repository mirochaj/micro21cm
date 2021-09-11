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

if __name__ == '__main__':
    test()
