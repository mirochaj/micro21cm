# micro21cm
model-independent constraints on reionization from observations of the 21-cm background

## Dependencies

You'll need `numpy`, `matplotlib`, `scipy`, `camb`, and `powerbox`.

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
ps = model.get_ps_21cm(z=8., k=k, Ts=3., Q=0.5, R_b=3., sigma_b=0.3)

# Plot dimensionless power spectrum
pl.loglog(k, k**3 * ps / 2. / np.pi**2)
```

Some built-in routines will plot things nicely, optionally with data, e.g.,

```python
# kwargs for the model
kw = {'z': 8., 'k': k, 'Ts': 3., 'Q': 0.5, 'R_b': 3., 'sigma_b': 0.3}

# Some made-up data
data = {8.: {'k': np.array([0.1, 0.2]),
    'D_sq': np.array([1e3, 2e3]), 'err': np.array([2e2, 2e2])}}

# Plot nicely.
fig, axes = micro21cm.plot_ps(model=model, model_kwargs=kw, data=data,
  fig_kwargs={'num': 2})
```
