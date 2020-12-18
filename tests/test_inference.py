"""

test_inference.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Wed 16 Dec 2020 21:31:49 EST

Description:

"""

import micro21cm
import numpy as np
import matplotlib.pyplot as pl

model = micro21cm.BubbleModel(bubbles=False, bubbles_ion=True)

# Some made-up data
data = {8.: {'k': np.array([0.1, 0.2]),
    'D_sq': np.array([1e3, 2e3]), 'err': np.array([2e2, 2e2])}}

fitter = micro21cm.FitMCMC(data, model=model)

sampler = fitter.run_fit(steps=100, nwalkers=256)

#micro21cm.plot_igm_constraints(sampler, model)

#for i in range(sampler.chain.shape[0]):
#    pl.plot(sampler.chain[i,:,0], color='b', alpha=0.1)


pl.hist(sampler.flatchain[:,0], bins=500, cumulative=True, density=True)

limits = micro21cm.get_limits_on_params(sampler, model)
print(limits)
