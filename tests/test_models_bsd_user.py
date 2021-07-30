"""

test_models_bsd_user.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Fri 30 Jul 2021 16:52:59 EDT

Description:

"""

import micro21cm
import numpy as np
import matplotlib.pyplot as pl

# Make a Gaussian BSD
R = np.logspace(-2, 3, 1000)
bsd = np.exp(-(R - 1.)**2 / 2. / 4.**2)

# Initialize model
model = micro21cm.BubbleModel(bubbles_pdf=(R, bsd))

k = np.arange(0.1, 1.1, 0.1)
ps = model.get_ps_21cm(z=8., k=k, Q=0.2, Ts=10)

pl.loglog(k, ps * k**3)
