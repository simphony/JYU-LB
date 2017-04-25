"""Poiseuille fluid flow simulation (flow between two parallel plates).

Details
-------
- 3D simulation (with one "dummy" direction)
- flow driven by pressure gradient
- periodic bc in the flow direction (as well as in the "dummy" direction)
- wall bc (immobile & adiabatic) for representing the plates

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

from jyulb.flow_field import poise_plate_steady_state_pgrad
# ---------------------------------------------------------------------------
# Fluid flow configuration
# ---------------------------------------------------------------------------
# Geometric setup, location of the plates
INLET  = FACE_1Z
OUTLET = FACE_Z1
UPPER_PLATE = FACE_Y1
LOWER_PLATE = FACE_1Y

PLATE_DIR = Y # The plates are separated in this direction
DUMMY_DIR = X # The "dummy" direction
FLOW_DIR  = Z # Fluid flow direction

# Lenghts and distances given in number of lattice nodes
L = 10 # Lenght of the domain in the flow direction
H = 11 # Distance between the plates
W = 5  # Width of the domain in the "dummy" direction

lat_size = np.zeros((3), dtype=np.int32)
lat_size[PLATE_DIR] = H
lat_size[DUMMY_DIR] = W
lat_size[FLOW_DIR]  = L

# Reference density, kinematic viscosity and relaxation scheme
rden = 1.0
kvisc = 2.0/6.0
rlx_scheme = BGK

# Simulation advanced by calling the evolution function
evol_period = 500  # time steps advanced per evolution function call
evol_call_cnt = 20 # number of evolution function calls to be executed

# Pressure gradient &
# Pressure difference between the first and last lattice site (flow dir.)
kinem_pgrad = -1e-5      # gradient of kinematic pressure 
pgrad = rden*kinem_pgrad # gradient of pressure
pdiff = (L-1.0)*pgrad    # outlet pressure - inlet pressure
pin_off = -0.5*pdiff     # inlet pressure (offset to the reference pressure)
pout_off = 0.5*pdiff     # outlet pressure (offset to the reference pressure)

# Gravity
g = np.zeros((3), dtype=np.float64)

# Boundary conditions
bc = np.full((6), PERIODIC, dtype=np.int32)
bc[INLET]  = FIXED_DEN
bc[OUTLET] = FIXED_DEN
bc[LOWER_PLATE] = WALL
bc[UPPER_PLATE] = WALL

# ---------------------------------------------------------------------------
# Set up the fluid flow solver
# ---------------------------------------------------------------------------
solver = Solver(lat_size, rlx_scheme, kvisc, g, bc)

phase = solver.get_phase() # phase field utilized by the solver
den = solver.get_den() # density field updated by the solver
vel = solver.get_vel() # velocity field updated by the solver

# Set phase information
phase[:,:,:] = FLUID

# Initialize the pressure field
pref = solver.eos_den2p(1.0) # reduced density = 1.0
pin  = pref + pin_off
pout = pref + pout_off

for index in np.ndindex(phase.shape):
    l = index[FLOW_DIR]
    p = pin + l*pgrad
    if phase[index] != SOLID:
        den[index] = solver.eos_p2den(p)

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
print 'Poiseuille flow simulation'
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
# Post-processing:
#   Analytical solution for the Poiseuille flow, error, and output
# ---------------------------------------------------------------------------
# GLE: flow field in y-data format
# ---------------------------------------------------------------------------
centrel_dist = np.zeros((H), dtype=np.float64)
ana_vel = np.zeros((H), dtype=np.float64)

dvisc = rden*kvisc
poise_plate_steady_state_pgrad(H, dvisc, pgrad, centrel_dist, ana_vel)

sim_loc_vel = np.zeros(3,dtype=np.float64)
coord = np.zeros(3,dtype=np.int32)
coord[X] = int(lat_size[X]/2)
coord[Y] = int(lat_size[Y]/2)
coord[Z] = int(lat_size[Z]/2)

ana2 = 0.0
sim_ana_diff2 = 0.0
sim_loc_vel = np.zeros((3), dtype=np.float64)

with open('flow_prof.txt', 'w') as f:
    for h in range(H):
        dist = centrel_dist[h]
        coord[PLATE_DIR] = h
        
        sim_loc_den = den[coord[X],coord[Y],coord[Z]]
        sim_loc_vel[X] = vel[coord[X],coord[Y],coord[Z],X]
        sim_loc_vel[Y] = vel[coord[X],coord[Y],coord[Z],Y]
        sim_loc_vel[Z] = vel[coord[X],coord[Y],coord[Z],Z]

        sim_loc_speed = np.sqrt(sim_loc_vel[X]*sim_loc_vel[X] +
                                sim_loc_vel[Y]*sim_loc_vel[Y] +
                                sim_loc_vel[Z]*sim_loc_vel[Z])

        ana_loc_vel = ana_vel[h]
        
        sim_ana_diff = sim_loc_vel[FLOW_DIR] - ana_loc_vel
        sim_ana_diff2 += sim_ana_diff*sim_ana_diff
        ana2 += ana_loc_vel*ana_loc_vel
        
        f.write('{0:f} {1:e} {2:e} {3:e} {4:e} {5:e} {6:e}\n'.format(dist,
                sim_loc_den, sim_loc_vel[X], sim_loc_vel[Y], sim_loc_vel[Z],
                sim_loc_speed, ana_loc_vel))

rel_l2_err_norm_vel = np.sqrt(sim_ana_diff2/ana2)

print 'Relative L2-error norm = ', rel_l2_err_norm_vel
print '-'*77
                
# ---------------------------------------------------------------------------
# Matlab: 2D flow field
# ---------------------------------------------------------------------------
coord[X] = int(lat_size[X]/2)
coord[Y] = int(lat_size[Y]/2)
coord[Z] = int(lat_size[Z]/2)

with open('flow_2D.field', 'w') as f:
    for x1 in range(lat_size[FLOW_DIR]):
        for x2 in range(lat_size[PLATE_DIR]):
            coord[FLOW_DIR] = x1
            coord[PLATE_DIR] = x2
            
            dn = den[coord[X],coord[Y],coord[Z]]
            v1 = vel[coord[X],coord[Y],coord[Z],FLOW_DIR]
            v2 = vel[coord[X],coord[Y],coord[Z],PLATE_DIR]
            
            f.write('{0:d} {1:d} {2:e} {3:e} {4:e}\n'.format(x1,x2,dn,v1,v2))
            
# ---------------------------------------------------------------------------
