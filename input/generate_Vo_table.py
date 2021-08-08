"""

generate_Vo_table.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Sun  8 Aug 2021 11:10:06 EDT

Description:

"""

import os
import h5py
import micro21cm
import numpy as np

model = micro21cm.BubbleModel()

fn = 'overlap_vol_log10R_{:.1f}_{:.1f}_N_{:.0f}.hdf5'.format(
    np.log10(model.Rmin), np.log10(model.Rmax), model.NR)

if os.path.exists(fn):
    raise IOError('{} exists!'.format(fn))

print("Will save overlap volume lookup table to {}".format(fn))

pb = micro21cm.util.ProgressBar(model.tab_R.size, name='Vo(d; R1, R2)')
pb.start()

tab = np.zeros([model.tab_R.size] * 3)
for h, d in enumerate(model.tab_R):
    pb.update(h)

    for i, R1 in enumerate(model.tab_R):
        tab[h,i,:] = np.array([model.get_overlap_vol_generic(d, R1, R2) \
                for j, R2 in enumerate(model.tab_R)])

pb.finish()

with h5py.File(fn, 'w') as f:
    f.create_dataset('Vo', data=tab)

print("Wrote {}.".format(fn))
