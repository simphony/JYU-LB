"""Lid-driven cavity flow (2D setup).

Details
-------
- 3D simulation (2D setup, i.e. periodic in one direction)
- flow driven by a moving wall

Author
------
Keijo Mattila, JYU, March 2017.
"""
# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
import numpy as np
from datetime import datetime, timedelta

from jyulb.isothermal import Solver
from jyulb.defs import FACE_1X,FACE_X1,FACE_1Y,FACE_Y1,FACE_1Z,FACE_Z1
from jyulb.defs import WALL,PERIODIC,FIXED_DEN,FIXED_VEL,SYMMETRIC
from jyulb.defs import BGK,TRT,REGULARIZATION
from jyulb.defs import FLUID,SOLID
from jyulb.defs import X,Y,Z

# ---------------------------------------------------------------------------
# Fluid flow configuration
# ---------------------------------------------------------------------------
# Geometric setup
UPPER_WALL = FACE_Z1 # moving wall
LOWER_WALL = FACE_1Z # wall opposite to the moving wall (not moving)
SIDE_WALL1 = FACE_1X # side wall 1, (not moving)
SIDE_WALL2 = FACE_X1 # side wall 2, (not moving)

MOVEW_NDIR = Z # Normal direction of the moving wall
SIDEW_NDIR = X # Normal direction of the side walls 1 and 2
PERIOD_DIR = Y # Direction of periodicity

# Lenghts and distances given in number of lattice nodes
H  = 100 # Distance between the lower and upper (moving) wall
SIDE_W   = 102  # Width of the domain in the side wall direction
PERIOD_W = 1  # Width of the domain in the periodic direction

# Moving wall velocity (in the side wall direction)
MWALL_VEL = 1.0*5e-2

lat_size = np.zeros((3), dtype=np.int32)
lat_size[MOVEW_NDIR] = H+2
lat_size[SIDEW_NDIR] = SIDE_W
lat_size[PERIOD_DIR] = PERIOD_W

# Reference density, kinematic viscosity and relaxation scheme
rden = 1.0
kvisc = 0.05/6.0
rlx_scheme = REGULARIZATION

# Simulation advanced by calling the evolution function
evol_period = 5000  # time steps advanced per evolution function call
evol_call_cnt = 25 # number of evolution function calls to be executed

# Gravity
g = np.zeros((3), dtype=np.float64)

# Boundary conditions
bc = np.full((6), PERIODIC, dtype=np.int32)

# Reynolds number
Re = H*MWALL_VEL/kvisc

# ---------------------------------------------------------------------------
# Set up the fluid flow solver
# ---------------------------------------------------------------------------
solver = Solver(lat_size, rlx_scheme, kvisc, g, bc)

phase = solver.get_phase() # phase field
den = solver.get_den() # density field updated by the solver
vel = solver.get_vel() # velocity field updated by the solver

# Set phase information
phase[:,:,:] = FLUID
pind = np.zeros((3), dtype=np.int32)

# Lower and upper wall
for index in np.ndindex((SIDE_W,PERIOD_W)):
    pind[SIDEW_NDIR] = index[0]
    pind[PERIOD_DIR] = index[1]

    # Lower wall (not moving)
    pind[MOVEW_NDIR] = 0
    phase[tuple(pind)] = SOLID
    
    # Upper wall (moving)
    pind[MOVEW_NDIR] = H+1
    phase[tuple(pind)] = SOLID
    vel[tuple(pind)+(SIDEW_NDIR,)] = MWALL_VEL

# Side walls
for index in np.ndindex((PERIOD_W,H)):
    pind[PERIOD_DIR] = index[0]
    pind[MOVEW_NDIR] = index[1]+1

    # Side wall 1 (not moving)
    pind[SIDEW_NDIR] = 0
    phase[tuple(pind)] = SOLID
    
    # Side wall 2 (not moving)
    pind[SIDEW_NDIR] = SIDE_W-1
    phase[tuple(pind)] = SOLID
    
# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------
flsites_cnt = solver.finalize_init()
flsites = solver.get_fluid_lat_sites()

tot_mass = 0
for fn in range(flsites_cnt):
    i = flsites[X,fn]
    j = flsites[Y,fn]
    k = flsites[Z,fn]
    tot_mass += den[i,j,k]
    
print '-'*77
print 'Lid-Driven Cavity flow simulation'
print '-'*77
print 'Reynolds number = {0:f}'.format(Re)
print '-'*77
print 'Total mass before simulation = {0:.6e}'.format(tot_mass)
print '-'*77

# Start timer
start_time = datetime.now()

tsteps = 0
for e in range(evol_call_cnt):
    tsteps = solver.evolve(evol_period)

    tot_vx = 0.0
    tot_vy = 0.0
    tot_vz = 0.0
    tot_mass = 0.0

    for fn in range(flsites_cnt):
        i = flsites[X,fn]
        j = flsites[Y,fn]
        k = flsites[Z,fn]
        tot_mass += den[i,j,k]
        tot_vx += vel[i,j,k,X]
        tot_vy += vel[i,j,k,Y]
        tot_vz += vel[i,j,k,Z]

    print 'Tstep = {0:6d}, tot.mass = {1:.6e}, tot.vel = ({2:13e}, {3:13e}, {4:13e})'.format(tsteps, tot_mass, tot_vx, tot_vy, tot_vz)
  
# Stop timer
end_time = datetime.now()
comp_time = end_time - start_time
dt = timedelta(days=comp_time.days, seconds=comp_time.seconds,
               microseconds=comp_time.microseconds)

# Compute MFLUPS               
mflu = flsites_cnt*tsteps/1.0e6
tots = dt.total_seconds()

if tots < 1e-6:
    mflups = 0.0
else:
    mflups = mflu/tots

tot_mass = 0
for fn in range(flsites_cnt):
    i = flsites[X,fn]
    j = flsites[Y,fn]
    k = flsites[Z,fn]
    tot_mass += den[i,j,k]
    
print '-'*77
print 'Total mass after simulation = {0:.6e}'.format(tot_mass)
print '-'*77
print 'Computing time (s) =', dt.total_seconds()
print 'MFLUPS =', mflups
print '-'*77

# ---------------------------------------------------------------------------
# Post-processing: output
# ---------------------------------------------------------------------------
# Matlab: 2D flow field
# ---------------------------------------------------------------------------
coord = np.zeros(3,dtype=np.int32)
coord[X] = int(lat_size[X]/2)
coord[Y] = int(lat_size[Y]/2)
coord[Z] = int(lat_size[Z]/2)

with open('flow_2D.field', 'w') as f:
    for x1 in range(lat_size[SIDEW_NDIR]):
        for x2 in range(lat_size[MOVEW_NDIR]):
            coord[SIDEW_NDIR] = x1
            coord[MOVEW_NDIR] = x2
            
            dn = den[coord[X],coord[Y],coord[Z]]
            v1 = vel[coord[X],coord[Y],coord[Z],SIDEW_NDIR]
            v2 = vel[coord[X],coord[Y],coord[Z],MOVEW_NDIR]
            
            f.write('{0:d} {1:d} {2:e} {3:e} {4:e}\n'.format(x1,x2,dn,v1,v2))
        
# ---------------------------------------------------------------------------
# GLE: vorticity in z-data format
# ---------------------------------------------------------------------------
# coord = np.zeros(3,dtype=np.int32)
# coord[X] = int(lat_size[X]/2)
# coord[Y] = int(lat_size[Y]/2)
# coord[Z] = int(lat_size[Z]/2)

# d1_forw = np.zeros(3,dtype=np.int32)
# d1_back = np.zeros(3,dtype=np.int32)
# d2_forw = np.zeros(3,dtype=np.int32)
# d2_back = np.zeros(3,dtype=np.int32)

# max_vort = -10.0
# min_vort = 10.0

# with open('vort_field.z', 'w') as f:
    # f.write('! nx {0:d} ny {1:d} xmin {2:d} xmax {3:d} ymin {4:d} ymax {5:d}\n'.format(SIDE_W-2, H, 0, SIDE_W-3, 0, H-1))

    # for x2 in range(H):
        # for x1 in range(SIDE_W-2):
            # coord[SIDEW_NDIR] = x1+1
            # coord[MOVEW_NDIR] = x2+1
            
            # d1_forw[:] = coord
            # d1_back[:] = coord
            # d2_forw[:] = coord
            # d2_back[:] = coord
            
            # d1_forw[SIDEW_NDIR] += 1
            # d1_back[SIDEW_NDIR] -= 1
            # d2_forw[MOVEW_NDIR] += 1
            # d2_back[MOVEW_NDIR] -= 1
            
            # d1_vel2 = 0.5*(vel[d1_forw[X],d1_forw[Y],d1_forw[Z],MOVEW_NDIR] - vel[d1_back[X],d1_back[Y],d1_back[Z],MOVEW_NDIR])
            # d2_vel1 = 0.5*(vel[d2_forw[X],d2_forw[Y],d2_forw[Z],SIDEW_NDIR] - vel[d2_back[X],d2_back[Y],d2_back[Z],SIDEW_NDIR])

            # vort = d1_vel2 - d2_vel1
            # f.write('{0:e} '.format(vort))
            
            # if vort > max_vort:
                # max_vort = vort
            # if vort < min_vort:
                # min_vort = vort

        # f.write('\n')

# print 'Max.vorticity = {0:.6e}, Min.vorticity = {1:.6e}'.format(max_vort,min_vort)
# print '-'*77
        
# ---------------------------------------------------------------------------
