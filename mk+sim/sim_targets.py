import numpy as np
import katpoint
from astropy import coordinates as c
from astropy import units as u

def skads_targs(targfile, center, model_ant, radius=1.0, fluxlim=0.000, maskfile=None, catfile=None):
    with open(targfile) as f:
        all_targs = f.read().splitlines()
    targets = []

    for targ in all_targs:
        this_target = katpoint.Target(targ)
        sep = np.rad2deg(this_target.separation(center, antenna=model_ant))
        if sep < radius and this_target.flux_density(1284.) > fluxlim:
            targets.append(this_target)
    print(f'Num Targets : {len(targets)}')
    if maskfile is not None:
        with open(maskfile, 'w') as f:
            for i in targets:
                coord = c.SkyCoord(*i.radec(), unit=(u.rad, u.rad), frame='icrs')
                strout = f'{coord.to_string("hmsdms", sep=" ", precision=1)} 10\n'
                f.write(strout)
    if catfile is not None:
        with open(catfile, 'w') as f:
            for i in targets:
                fm=i.flux_model.coefs[0:4]
                coord = c.SkyCoord(*i.radec(), unit=(u.rad, u.rad), frame='icrs')
                flux = i.flux_density(1284.)
                strout = f'{coord.to_string("hmsdms", sep=" ", precision=1)} {flux*1e3:7.3f}  {fm[0]:7.3f} {fm[1]:7.3f} {fm[2]: .3e} {fm[3]: .3e}\n'
                f.write(strout)
    return targets

def grid_targs(maskfile=None, catfile=None):
    targets = [katpoint.Target("A, radec, 00:08:00.0, -79:40:00.0, (1.0 1.0e6 1.15 -0.7 0.0 0.0 0.0 0.0 1.0 0.3 0.01 0.0)"),
               katpoint.Target("B, radec, 00:00:00.0, -79:40:00.0, (1.0 1.0e6 1.15 -0.7 0.0 0.0 0.0 0.0 1.0 0.3 0.1 0.0)"),
               katpoint.Target("C, radec, 23:52:00.0, -79:40:00.0, (1.0 1.0e6 1.15 -0.7 0.0 0.0 0.0 0.0 1.0 0.1 0.3 0.0)"),
               katpoint.Target("D, radec, 00:08:00.0, -80:00:00.0, (1.0 1.0e6 1.15 -0.7 0.0 0.0 0.0 0.0 1.0 0.1 0.3 0.5)"),
               katpoint.Target("E, radec, 00:00:00.0, -80:00:00.0, (1.0 1.0e6 1.15 -0.7 0.0 0.0 0.0 0.0 1.0 0.1 0.01 0.001)"),
               katpoint.Target("F, radec, 23:52:00.0, -80:00:00.0, (1.0 1.0e6 1.15 -0.7 0.0 0.0 0.0 0.0 1.0 0.01 0.1 0.5)"),
               katpoint.Target("G, radec, 00:08:00.0, -80:20:00.0, (1.0 1.0e6 1.15 -0.7 0.0 0.0 0.0 0.0 1.0 0.01 0.3 0.001)"),
               katpoint.Target("H, radec, 00:00:00.0, -80:20:00.0, (1.0 1.0e6 1.15 -0.7 0.0 0.0 0.0 0.0 1.0 0.3 0.1 0.5)"),
               katpoint.Target("I, radec, 23:52:00.0, -80:20:00.0, (1.0 1.0e6 1.15 -0.7 0.0 0.0 0.0 0.0 1.0 0.3 0.1 0.001)")]

    if maskfile is not None:
        with open(maskfile, 'w') as f:
            for i in targets:
                coord = c.SkyCoord(*i.radec(), unit=(u.rad, u.rad), frame='icrs')
                strout = f'{coord.to_string("hmsdms", sep=" ", precision=1)} 10\n'
                f.write(strout)
    if catfile is not None:
        with open(catfile, 'w') as f:
            for i in targets:
                coord = c.SkyCoord(*i.radec(), unit=(u.rad, u.rad), frame='icrs')
                flux = i.flux_density(1284.)
                strout = f'{coord.to_string("hmsdms", sep=" ", precision=1)} {flux*1e3:7.3f}\n'
                f.write(strout)
    return targets
