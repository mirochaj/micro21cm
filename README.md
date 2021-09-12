# micro21cm

model-independent constraints on reionization from observations of the 21-cm background

![Tests](https://github.com/mirochaj/micro21cm/actions/workflows/test_suite.yaml/badge.svg) [![codecov](https://codecov.io/gh/mirochaj/micro21cm/branch/main/graph/badge.svg?token=18ZMZEUWPW)](https://codecov.io/gh/mirochaj/micro21cm)

## Dependencies

You'll need `numpy`, `matplotlib`, `scipy`, `camb`, and optionally, `powerbox`. If you want to run fits, you'll need `emcee`, and if you want to do so in parallel, you'll need `mpi4py` and `schwimmbad` for MPI parallelism. Alternatively, you can use `multiprocessing` on shared memory machines.

## Quick Example

To plot a simple bubble model for the 21-cm power spectrum, you can do something
like:

```python
import micro21cm
import numpy as np
import matplotlib.pyplot as pl

model = micro21cm.BubbleModel()

# Set modes of interest
k = np.logspace(-1., 0, 21)

# Compute P(k)
ps = model.get_ps_21cm(z=8., k=k, Ts=3., Q=0.5, R=3., sigma=0.6)

# Plot dimensionless power spectrum
pl.loglog(k, k**3 * ps / 2. / np.pi**2)
```
