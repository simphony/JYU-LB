"""Couette fluid flow simulation (flow between two moving, parallel plates).

Details
-------
- 3D simulation (with two "dummy" directions)
- flow driven by the moving plates
- periodic bc in the "dummy" directions

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

from jyulb.flow_field import couette_plate_steady_state

# ---------------------------------------------------------------------------
# Fluid flow configuration
# ---------------------------------------------------------------------------
# Geometric setup, location of the plates
PLATE_DIR = X # The plates are separated in this direction
DUMM1_DIR = Y # The "dummy" direction 1
DUMM2_DIR = Z # The "dummy" direction 2

# Lenghts and distances given in number of lattice nodes
H  = 10 # Distance between the plates
W1 = 5  # Width of the domain in the "dummy" direction 1
W2 = 4  # Width of the domain in the "dummy" direction 2

# Plate (or wall) velocities
LOWER_PLATE_DUMM1_DIR_VEL = 0.0*1e-5
LOWER_PLATE_DUMM2_DIR_VEL = 0.0*1e-5

UPPER_PLATE_DUMM1_DIR_VEL = 1.0*1e-5
UPPER_PLATE_DUMM2_DIR_VEL = 2.0*1e-5

lat_size = np.zeros((3), dtype=np.int32)
lat_size[PLATE_DIR] = H+2
lat_size[DUMM1_DIR] = W1
lat_size[DUMM2_DIR] = W2

# Reference density, kinematic viscosity and relaxation scheme
rden = 1.0
kvisc = 0.5/6.0
rlx_scheme = BGK

# Simulation advanced by calling the evolution function
evol_period = 500  # time steps advanced per evolution function call
evol_call_cnt = 20 # number of evolution function calls to be executed

# Gravity
g = np.zeros((3), dtype=np.float64)

# Boundary conditions
bc = np.full((6), PERIODIC, dtype=np.int32)

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

for index in np.ndindex((W1,W2)):
    pind[DUMM1_DIR] = index[0]
    pind[DUMM2_DIR] = index[1]

    # Lower plate
    pind[PLATE_DIR] = 0
    phase[tuple(pind)] = SOLID
    vel[tuple(pind)+(DUMM1_DIR,)] = LOWER_PLATE_DUMM1_DIR_VEL
    vel[tuple(pind)+(DUMM2_DIR,)] = LOWER_PLATE_DUMM2_DIR_VEL
    
    # Upper plate
    pind[PLATE_DIR] = H+1
    phase[tuple(pind)] = SOLID
    vel[tuple(pind)+(DUMM1_DIR,)] = UPPER_PLATE_DUMM1_DIR_VEL
    vel[tuple(pind)+(DUMM2_DIR,)] = UPPER_PLATE_DUMM2_DIR_VEL

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
print 'Couette flow simulation'
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
#   Analytical solution for the Couette flow, error, and output
# ---------------------------------------------------------------------------
# GLE: flow field in y-data format
# ---------------------------------------------------------------------------
centrel_dist = np.zeros((H), dtype=np.float64)
ana_vel = np.zeros((H,2), dtype=np.float64)
couette_plate_steady_state(H, LOWER_PLATE_DUMM1_DIR_VEL,
                           LOWER_PLATE_DUMM2_DIR_VEL,
                           UPPER_PLATE_DUMM1_DIR_VEL,
                           UPPER_PLATE_DUMM2_DIR_VEL,
                           centrel_dist, ana_vel)

sim_loc_vel = np.zeros(3,dtype=np.float64)
coord = np.zeros(3,dtype=np.int32)
coord[X] = int(lat_size[X]/2)
coord[Y] = int(lat_size[Y]/2)
coord[Z] = int(lat_size[Z]/2)

ana2 = 0.0
sim_ana_diff2 = 0.0
sim_loc_vel = np.zeros((3), dtype=np.float64)

with open('flow_prof.txt', 'w') as f:
    for h in range(1,H+1):
        d = centrel_dist[h-1]
        ana_dumm1_vel = ana_vel[h-1,0]
        ana_dumm2_vel = ana_vel[h-1,1]
        coord[PLATE_DIR] = h
        
        sim_loc_den = den[coord[X],coord[Y],coord[Z]]
        sim_loc_vx = vel[coord[X],coord[Y],coord[Z],X]
        sim_loc_vy = vel[coord[X],coord[Y],coord[Z],Y]
        sim_loc_vz = vel[coord[X],coord[Y],coord[Z],Z]

        sim_loc_speed = np.sqrt(sim_loc_vx*sim_loc_vx +
                                sim_loc_vy*sim_loc_vy +
                                sim_loc_vz*sim_loc_vz)

        ana_loc_speed = np.sqrt(ana_dumm1_vel*ana_dumm1_vel +
                                ana_dumm2_vel*ana_dumm2_vel)

        sim_ana_diff = sim_loc_speed - ana_loc_speed
        sim_ana_diff2 += sim_ana_diff*sim_ana_diff
        ana2 += ana_loc_speed*ana_loc_speed
        
        f.write('{0:f} {1:e} {2:e} {3:e} {4:e} {5:e} {6:e}\n'.format(d,
                sim_loc_den, sim_loc_vx, sim_loc_vy, sim_loc_vz,
                sim_loc_speed, ana_loc_speed))

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
    for x1 in range(lat_size[DUMM1_DIR]):
        for x2 in range(lat_size[PLATE_DIR]):
            coord[DUMM1_DIR] = x1
            coord[PLATE_DIR] = x2
            
            dn = den[coord[X],coord[Y],coord[Z]]
            v1 = vel[coord[X],coord[Y],coord[Z],DUMM1_DIR]
            v2 = vel[coord[X],coord[Y],coord[Z],PLATE_DIR]
            
            f.write('{0:d} {1:d} {2:e} {3:e} {4:e}\n'.format(x1,x2,dn,v1,v2))
            
# ---------------------------------------------------------------------------
