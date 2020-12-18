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

model_lognorm = micro21cm.BubbleModel()
model_plexp = micro21cm.BubbleModel(bubbles_pdf='plexp')

bsd_log = model_lognorm.get_bsd(Q=0.1, sigma_b=0.5)
bsd_ple = model_plexp.get_bsd(Q=0.1, sigma_b=1.) # sigma_b = PL index

fig, ax = pl.subplots(1, 1)

ax.loglog(model_lognorm.tab_R, bsd_log)
ax.loglog(model_plexp.tab_R, bsd_ple)

ax.set_ylim(1e-8, 1e-1)
