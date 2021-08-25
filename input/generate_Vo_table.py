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

fn3d = 'overlap_vol_log10R_{:.1f}_{:.1f}_N_{:.0f}_3D.hdf5'.format(
    np.log10(model.Rmin), np.log10(model.Rmax), model.NR)
fn2d = 'overlap_vol_log10R_{:.1f}_{:.1f}_N_{:.0f}_2D.hdf5'.format(
    np.log10(model.Rmin), np.log10(model.Rmax), model.NR)

if os.path.exists(fn2d):
    raise IOError('{} exists!'.format(fn))

print("Will save overlap volume lookup tables to {} and {}".format(fn3d, fn2d))

pb = micro21cm.util.ProgressBar(model.tab_R.size, name='Vo(d; R1, R2)')
pb.start()

tab2d = np.zeros([model.tab_R.size])
tab3d = np.zeros([model.tab_R.size] * 3)
for h, d in enumerate(model.tab_R):
    pb.update(h)

    # Could just retrieve diagonal from 3-D table but I'm lazy
    tab2d[h,:] = np.array([model.get_overlap_vol(d, R) \
            for j, R in enumerate(model.tab_R)])

    for i, R1 in enumerate(model.tab_R):
        tab3d[h,i,:] = np.array([model.get_overlap_vol_generic(d, R1, R2) \
                for j, R2 in enumerate(model.tab_R)])

pb.finish()

with h5py.File(fn2d, 'w') as f:
    f.create_dataset('Vo', data=tab2d)

print("Wrote {}.".format(fn2d))

with h5py.File(fn3d, 'w') as f:
    f.create_dataset('Vo', data=tab3d)

print("Wrote {}.".format(fn3d))
