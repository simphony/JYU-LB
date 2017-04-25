"""Lattice-Boltzmann solver for the simulation of
isothermal (single-phase) fluid flows.

Details
-------
The solver assumes/operates with non-dimensional variables/parameters.

Specifically, let DR and DT denote lattice spacing and discrete time step,
respectively. Then CR = DR/DT is the so-called lattice speed and, e.g.,
local flow velocity is always expressed in the non-dimensional form U/CR
(where U is the local flow velocity in SI units).

Moreover, the local (mass) density is always treated as the reduced density
DEN/RDEN (where both the local density DEN and the reference density RDEN
are in SI units).

To summarize, the chosen system of reduced (or lattice) units involves
three parameters

  - length  DR   (lattice spacing, SI units),
  - time    DT   (discrete time step, SI units),
  - density RDEN (reference density, SI units).
  
Note that the solver does not transform units (using the params. DR,DT,RDEN);
the solver relies on external unit transformation - important!!!
                                                  ==============
                                                           
In other words, all variables/parameters are treated in
non-dimensional/reduced form, e.g.

  - length         L* = L/DR          (L in SI units),
  - time           T* = T/DT          (T in SI units),
  - velocity       V* = V/CR          (V in SI units),
  - mass density DEN* = DEN/RDEN      (DEN in SI units),
  - pressure       P* = P/(RDEN*CR^2) (P in SI units),
  - kinem.visc.   nu* = nu/(DT*CR^2)  (nu in SI units).
  
Python interface
----------------
Constructors (2):
-----------------
  Solver(int[:] lat_size not None, int rlx_scheme = BGK,
         double kvisc = 1.0/6.0, double[:] g = None, int[:] bc = None)
  - the default constructor for creating a solver
    (without prescribing the initial flow field)
         
  @classmethod 
  with_iflow(unsigned char[:, :, ::1] phase not None,
             double[:, :, ::1] iden not None,
             double[:, :, :, ::1] ivel not None,
             int rlx_scheme = BGK, double kvisc = 1.0/6.0,
             double[:] g = None, int[:] bc = None)
  - constructor for creating a solver with initial flow field prescribed

Properties (1): 
---------------
  gravity (get/set)
  - vector (3 components, float)
  
Update methods (2): 
-------------------
  unsigned int finalize_init()
  - finalizes the solver initialization  
  - returns the number of fluid lattice sites in the simulation domain
  - must be called before the first call to the evolve method (see below)
  - raises an exception (RuntimeError)
    if error has occured in the solver setup
   
  unsigned int evolve(unsigned int tsteps)
  - execute the given number of time steps
    (i.e. update distributions values at the fluid lattice sites).
  - returns the total number of time steps executed (since initialization)
  - raises an exception (RuntimeError)
    if the solver initialization is not finalized
    (i.e. the finalize_init method has not been called)
    or
    if error has occured in the solver setup
        
Status check methods (1): 
-------------------------
  int error_occured(self)
  - returns 0 if no error has occured (and a non-zero value otherwise)
  - the readonly class attribute error_msg (string) reports the error

Getter methods (4 + 6 + 6 + 1): 
-------------------------------
  unsigned char[:, :, ::1] get_phase()
  - returns the phase field (as a typed memoryview)
    utilized (not updated) by the solver

  double[:, :, :, ::1] get_acc()
  - returns the external acceleration field (as a typed memoryview)
    utilized (not updated) by the solver

  double[:, :, :, ::1] get_vel()
  - returns the velocity field (as a typed memoryview)
    updated by the solver 

  double[:, :, ::1] get_den()
  - returns the (mass) density field (as a typed memoryview)
    updated by the solver 

  unsigned int[:, ::1] get_fluid_lat_sites()
  - returns the index coordinates of fluid lattice sites in the simul.domain
  - fluid_lat_site[X,0] = 1st index.coord (i) of the 1st fluid lat.site
  
Constitutive equations/material relations (2): 
----------------------------------------------
  double eos_den2p(double den)
  - pressure as a function of density (equation of state)

  double eos_p2den(double p)
  - density as a function of pressure (equation of state)
 
Author
------
Keijo Mattila, JYU, April 2017.
"""
# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
import cython
cimport cython
import numpy as np
cimport numpy as np

from cython.parallel cimport prange, parallel, threadid
cimport openmp

from cython cimport boundscheck, wraparound, cdivision
from libc.stdlib cimport malloc, free

cimport bcs
from defs cimport X,Y,Z
from defs cimport FLUID, SOLID
from defs cimport FACE_1X, FACE_X1, FACE_1Y, FACE_Y1, FACE_1Z, FACE_Z1
from defs cimport WALL, PERIODIC, FIXED_DEN, FIXED_VEL,SYMMETRIC
from defs cimport BGK, TRT, REGULARIZATION

from D3Q19 cimport L_1I_1J_K0, L_1I_J0_1K, L_1I_J0_K0, L_1I_J0_K1, L_1I_J1_K0
from D3Q19 cimport L_I0_1J_1K, L_I0_1J_K0, L_I0_1J_K1
from D3Q19 cimport L_I0_J0_1K, L_I0_J0_K0, L_I0_J0_K1
from D3Q19 cimport L_I0_J1_1K, L_I0_J1_K0, L_I0_J1_K1
from D3Q19 cimport L_I1_1J_K0, L_I1_J0_1K, L_I1_J0_K0, L_I1_J0_K1, L_I1_J1_K0
from D3Q19 cimport H2_XX, H2_XY, H2_XZ, H2_YY, H2_YZ, H2_ZZ
from D3Q19 cimport HMOMS

# ---------------------------------------------------------------------------
# Class Solver
# ---------------------------------------------------------------------------
cdef class Solver:

    # -----------------------------------------------------------------------
    # Constructors
    # -----------------------------------------------------------------------
    def __cinit__(self, int[:] lat_size not None, int rlx_scheme = BGK,
                  double kvisc = 1.0/6.0, double[:] g = None,
                  int[:] bc = None):
        """The default constructor for creating a solver
        (without prescribing the initial flow field).
        
        Parameters
        ----------
        lat_size : int[:] (typed memoryview, element count >= 3, read)
                   size of the lattice (i.e. the number of lattice sites)
        rlx_scheme : int (default BGK)
                     relaxation scheme (i.e. a particular collision operator)
        kvisc : double
                kinematic viscosity (non-dimensional, default 1/6)
        g : double[:] (typed memoryview, element count >= 3, read)
            gravity (non-dimensional, default None -> (0,0,0) )
        bc : int[:] (typed memoryview, element count >= 6, read)
             bound.conds.per domain face (default None -> fully periodic)

        Raises
        ------
        ValueError
           if the lattice size is not >= 1 (in each lattice direction)
        ValueError
           if rlx_scheme is not valid
        ValueError
           if kvisc <= 0.0
        ValueError
           if the configuration of boundary conditions is not allowed
        """
        cdef:
            double tau_e, inv_tau_e, inv_tau_o
            int l, li, lj, lk
            int NI, NJ, NK

        self.error_msg = ''
            
        # Check arguments
        valid_rlx_scheme = (BGK, TRT, REGULARIZATION)

        if lat_size[X] < 1 or lat_size[Y] < 1 or lat_size[Z] < 1:
            self.error_msg = 'Invalid lattice size input.'
            raise ValueError(self.error_msg)

        if rlx_scheme not in valid_rlx_scheme:
            self.error_msg = 'Invalid relaxation scheme input.'
            raise ValueError(self.error_msg)
            
        if kvisc <= 0.0:
            self.error_msg = 'Invalid kinematic viscosity input.'
            raise ValueError(self.error_msg)

        if g is None:
            self.g[X] = 0.0
            self.g[Y] = 0.0
            self.g[Z] = 0.0
        else:
            self.g[X] = g[X]
            self.g[Y] = g[Y]
            self.g[Z] = g[Z]

        if bc is None:
            self.bc[FACE_1X] = PERIODIC
            self.bc[FACE_X1] = PERIODIC
            self.bc[FACE_1Y] = PERIODIC
            self.bc[FACE_Y1] = PERIODIC
            self.bc[FACE_1Z] = PERIODIC
            self.bc[FACE_Z1] = PERIODIC
        else:
            self._check_bc(bc)
            self.bc[FACE_1X] = bc[FACE_1X]
            self.bc[FACE_X1] = bc[FACE_X1]
            self.bc[FACE_1Y] = bc[FACE_1Y]
            self.bc[FACE_Y1] = bc[FACE_Y1]
            self.bc[FACE_1Z] = bc[FACE_1Z]
            self.bc[FACE_Z1] = bc[FACE_Z1]

        # Lattice size, add halo layers
        NI = lat_size[X]+2
        NJ = lat_size[Y]+2
        NK = lat_size[Z]+2
        self.NI = NI
        self.NJ = NJ
        self.NK = NK
        self.NCNT = NI*NJ*NK

        # Flow field, hydrodynamic variables
        self.phase = np.full((NI,NJ,NK),SOLID,dtype=np.uint8)
        self.fluid_lsite_domain = None
        
        self.den = np.ones((NI,NJ,NK),dtype=np.float64)
        self.vel = np.zeros((NI,NJ,NK,3),dtype=np.float64)
        self.acc = np.zeros((NI,NJ,NK,3),dtype=np.float64)

        # Boundary conditions (and related data)
        self._init_bc_scheme()

        self.bc_data_den_face_1X = np.ones((NJ,NK),dtype=np.float64)
        self.bc_data_den_face_X1 = np.ones((NJ,NK),dtype=np.float64)
        self.bc_data_den_face_1Y = np.ones((NI,NK),dtype=np.float64)
        self.bc_data_den_face_Y1 = np.ones((NI,NK),dtype=np.float64)
        self.bc_data_den_face_1Z = np.ones((NI,NJ),dtype=np.float64)
        self.bc_data_den_face_Z1 = np.ones((NI,NJ),dtype=np.float64)

        self.bc_data_vel_face_1X = np.zeros((NJ,NK,3),dtype=np.float64)
        self.bc_data_vel_face_X1 = np.zeros((NJ,NK,3),dtype=np.float64)
        self.bc_data_vel_face_1Y = np.zeros((NI,NK,3),dtype=np.float64)
        self.bc_data_vel_face_Y1 = np.zeros((NI,NK,3),dtype=np.float64)
        self.bc_data_vel_face_1Z = np.zeros((NI,NJ,3),dtype=np.float64)
        self.bc_data_vel_face_Z1 = np.zeros((NI,NJ,3),dtype=np.float64)

        # Discrete velocity set and distribution functions
        self.dvs = D3Q19_H10()
        self.f1 = np.zeros(self.NCNT*LVECS, dtype=np.float64)
        self.f2 = np.zeros(self.NCNT*LVECS, dtype=np.float64)
        
        # Relaxation scheme (and related parameters)
        self.kvisc = kvisc
        tau_e = 0.5 + self.dvs.AS2*kvisc
        inv_tau_e = 1.0/tau_e

        self.rlx_scheme = rlx_scheme
        self.rlx_prm[0] = inv_tau_e

        if rlx_scheme == BGK:
            self.lbe_func = self.dvs.forw_euler_BGK
        elif rlx_scheme == TRT:
            self.lbe_func = self.dvs.forw_euler_TRT
            inv_tau_o = 8.0*(2.0-inv_tau_e)/(8.0-inv_tau_e)
            self.rlx_prm[1] = inv_tau_o
        else:
            self.lbe_func = self.dvs.forw_euler_regul

        # Auxiliarry array for the iteration of lattice sites
        for l in range(LVECS):
            li = self.dvs.LI[l]
            lj = self.dvs.LJ[l]
            lk = self.dvs.LK[l]
            self.njump[l] = self._N_IJK(li,lj,lk)

        # Declaration of the solver state (i.e. not yet fully initialized)
        self.tot_tsteps = 0
        self.init_done = 0

    # -----------------------------------------------------------------------
    @classmethod
    def with_iflow(cls, unsigned char[:, :, ::1] phase not None,
                   double[:, :, ::1] iden not None,
                   double[:, :, :, ::1] ivel not None,
                   int rlx_scheme = BGK, double kvisc = 1.0/6.0,
                   double[:] g = None, int[:] bc = None):
        """Constructor for creating a solver
        with initial flow field prescribed.
        
        Parameters
        ----------
        phase : unsigned char[:, :, ::1] (typed memoryview, read)
                phase information for the lattice sites
        iden : double[:, :, ::1] (typed memoryview, read)
               initial density field
        ivel : double[:, :, :, ::1] (typed memoryview, read)
               initial velocity field
               ivel[0,0,0,X] = velocity (x-comp.) for the lat.site (0,0,0)
        rlx_scheme : int (default BGK)
                     relaxation scheme (i.e. a particular collision operator)
        kvisc : double
                kinematic viscosity (non-dimensional, default 1/6)
        g : double[:] (typed memoryview, element count >= 3, read)
            gravity (non-dimensional, default None -> (0,0,0) )
        bc : int[:] (typed memoryview, element count >= 6, read)
             bound.conds.per domain face (default None -> fully periodic)

        Raises
        ------
        ValueError
           if the sizes of phase, iden, and ivel do not match
        """
        cdef:
            int psx = phase.shape[X]
            int psy = phase.shape[Y]
            int psz = phase.shape[Z]
            str err_msg
            Solver slvr

        if iden.shape[X] != psx or ivel.shape[X] != psx:
            err_msg = 'Invalid input for the initial flow field: '
            err_msg += 'array sizes do not match (x-dimension)!'
            raise ValueError(err_msg)
        if iden.shape[Y] != psy or ivel.shape[Y] != psy:
            err_msg = 'Invalid input for the initial flow field: '
            err_msg += 'array sizes do not match (y-dimension)!'
            raise ValueError(err_msg)
        if iden.shape[Z] != psz or ivel.shape[Z] != psz:
            err_msg = 'Invalid input for the initial flow field: '
            err_msg += 'array sizes do not match (z-dimension)!'
            raise ValueError(err_msg)

        slvr = Solver(np.array(phase.shape, dtype=np.int32),
                      rlx_scheme, kvisc, g, bc)

        slvr.phase[1:slvr.NI-1,1:slvr.NJ-1,1:slvr.NK-1] = phase
        slvr.den[1:slvr.NI-1,1:slvr.NJ-1,1:slvr.NK-1] = iden
        slvr.vel[1:slvr.NI-1,1:slvr.NJ-1,1:slvr.NK-1,:] = ivel

        return slvr

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------
    @property
    def gravity(self):
        return self.g
        
    @gravity.setter
    def gravity(self, g):
        self.g[X] = g[X]
        self.g[Y] = g[Y]
        self.g[Z] = g[Z]
    
    # -----------------------------------------------------------------------
    # Configuration of the boundary conditions (private methods)
    # -----------------------------------------------------------------------
    cdef void _check_bc(self, int[:] bc):
        """Check the configuration of boundary conditions.
        """
        cdef:
            int s

        valid_bc = (WALL,PERIODIC,FIXED_DEN,FIXED_VEL,SYMMETRIC)
        fixed_var_bc = (FIXED_DEN,FIXED_VEL)
    
        for s in range(6):
            if bc[s] not in valid_bc:
                self.error_msg = 'Invalid boundary condition input.'
                raise ValueError(self.error_msg)

        # Check consistency of periodicity configurations
        if ( (bc[FACE_1X] == PERIODIC or bc[FACE_X1] == PERIODIC) and
             (bc[FACE_1X] != bc[FACE_X1]) ):
            self.error_msg = 'Invalid periodic bc setup (in x-dir.).'
            raise ValueError(self.error_msg)
                
        if ( (bc[FACE_1Y] == PERIODIC or bc[FACE_Y1] == PERIODIC) and
             (bc[FACE_1Y] != bc[FACE_Y1]) ):
            self.error_msg = 'Invalid periodic bc setup (in y-dir.).'
            raise ValueError(self.error_msg)
              
        if ( (bc[FACE_1Z] == PERIODIC or bc[FACE_Z1] == PERIODIC) and
             (bc[FACE_1Z] != bc[FACE_Z1]) ):
            self.error_msg = 'Invalid periodic bc setup (in z-dir.).'
            raise ValueError(self.error_msg)

        # Fixed variable bcs for two faces sharing an edge not allowed
        # (i.e. fixed variable bcs can be configured for opposite faces only)
        if ( (bc[FACE_1X] in fixed_var_bc or bc[FACE_X1] in fixed_var_bc) and
             (bc[FACE_1Y] in fixed_var_bc or bc[FACE_Y1] in fixed_var_bc or
              bc[FACE_1Z] in fixed_var_bc or bc[FACE_Z1] in fixed_var_bc) ):
            self.error_msg = 'Invalid fixed variable bc setup.'
            raise ValueError(self.error_msg)

        if ( (bc[FACE_1Y] in fixed_var_bc or bc[FACE_Y1] in fixed_var_bc) and
             (bc[FACE_1X] in fixed_var_bc or bc[FACE_X1] in fixed_var_bc or
              bc[FACE_1Z] in fixed_var_bc or bc[FACE_Z1] in fixed_var_bc) ):
            self.error_msg = 'Invalid fixed variable bc setup.'
            raise ValueError(self.error_msg)
        
        if ( (bc[FACE_1Z] in fixed_var_bc or bc[FACE_Z1] in fixed_var_bc) and
             (bc[FACE_1X] in fixed_var_bc or bc[FACE_X1] in fixed_var_bc or
              bc[FACE_1Y] in fixed_var_bc or bc[FACE_Y1] in fixed_var_bc) ):
            self.error_msg = 'Invalid fixed variable bc setup.'
            raise ValueError(self.error_msg)
        
    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cdef void _init_bc_scheme(self):
        """Store appropriate boundary condition functions.
        """
        cdef int NI = self.NI, NJ = self.NJ, NK = self.NK
        
        self.bc_func[FACE_1X] = NULL
        self.bc_func[FACE_X1] = NULL
        self.bc_func[FACE_1Y] = NULL
        self.bc_func[FACE_Y1] = NULL
        self.bc_func[FACE_1Z] = NULL
        self.bc_func[FACE_Z1] = NULL
        
        if self.bc[FACE_1X] == FIXED_DEN:
            self.bc_func[FACE_1X] = bcs.fix_den_face_1x_D3Q19
        elif self.bc[FACE_1X] == FIXED_VEL:
            self.bc_func[FACE_1X] = bcs.fix_vel_face_1x_D3Q19
        
        if self.bc[FACE_X1] == FIXED_DEN:
            self.bc_func[FACE_X1] = bcs.fix_den_face_x1_D3Q19
        elif self.bc[FACE_X1] == FIXED_VEL:
            self.bc_func[FACE_X1] = bcs.fix_vel_face_x1_D3Q19
        
        if self.bc[FACE_1Y] == FIXED_DEN:
            self.bc_func[FACE_1Y] = bcs.fix_den_face_1y_D3Q19
        elif self.bc[FACE_1Y] == FIXED_VEL:
            self.bc_func[FACE_1Y] = bcs.fix_vel_face_1y_D3Q19
        
        if self.bc[FACE_Y1] == FIXED_DEN:
            self.bc_func[FACE_Y1] = bcs.fix_den_face_y1_D3Q19
        elif self.bc[FACE_Y1] == FIXED_VEL:
            self.bc_func[FACE_Y1] = bcs.fix_vel_face_y1_D3Q19
        
        if self.bc[FACE_1Z] == FIXED_DEN:
            self.bc_func[FACE_1Z] = bcs.fix_den_face_1z_D3Q19
        elif self.bc[FACE_1Z] == FIXED_VEL:
            self.bc_func[FACE_1Z] = bcs.fix_vel_face_1z_D3Q19
        
        if self.bc[FACE_Z1] == FIXED_DEN:
            self.bc_func[FACE_Z1] = bcs.fix_den_face_z1_D3Q19
        elif self.bc[FACE_Z1] == FIXED_VEL:
            self.bc_func[FACE_Z1] = bcs.fix_vel_face_z1_D3Q19
    
    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cdef void _set_bc_data(self):
        """Boundary condition data set by copying values
        from the density and velocity fields.
        """
        if self.bc[FACE_1X] == FIXED_DEN or self.bc[FACE_1X] == FIXED_VEL:
            self.bc_data_den_face_1X[:,::1] = self.den[1,:,:]
            self.bc_data_vel_face_1X[:,:,::1] = self.vel[1,:,:,:]
            
        if self.bc[FACE_X1] == FIXED_DEN or self.bc[FACE_X1] == FIXED_VEL:
            self.bc_data_den_face_X1[:,::1] = self.den[self.NI-2,:,:]
            self.bc_data_vel_face_X1[:,:,::1] = self.vel[self.NI-2,:,:,:]
            
        if self.bc[FACE_1Y] == FIXED_DEN or self.bc[FACE_1Y] == FIXED_VEL:
            self.bc_data_den_face_1Y[:,::1] = self.den[:,1,:]
            self.bc_data_vel_face_1Y[:,:,::1] = self.vel[:,1,:,:]

        if self.bc[FACE_Y1] == FIXED_DEN or self.bc[FACE_Y1] == FIXED_VEL:
            self.bc_data_den_face_Y1[:,::1] = self.den[:,self.NJ-2,:]
            self.bc_data_vel_face_Y1[:,:,::1] = self.vel[:,self.NJ-2,:,:]
            
        if self.bc[FACE_1Z] == FIXED_DEN or self.bc[FACE_1Z] == FIXED_VEL:
            self.bc_data_den_face_1Z[:,::1] = self.den[:,:,1]
            self.bc_data_vel_face_1Z[:,:,::1] = self.vel[:,:,1,:]
            
        if self.bc[FACE_Z1] == FIXED_DEN or self.bc[FACE_1Z] == FIXED_VEL:
            self.bc_data_den_face_Z1[:,::1] = self.den[:,:,self.NK-2]
            self.bc_data_vel_face_Z1[:,:,::1] = self.vel[:,:,self.NK-2,:]
            
    # -----------------------------------------------------------------------
    # Enforcement of the boundary conditions (private methods)
    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cdef void _enforce_phase_per_sym_bc(self):
        """Update the phase field according to the boundary conditions,
        i.e. set phase information for the halo lattice sites.
        """
        cdef int i, j, k, NI = self.NI, NJ = self.NJ, NK = self.NK

        # Set phase for the halo lattice sites, boundary face 1X
        if ( self.bc[FACE_1X] == PERIODIC or
             self.bc[FACE_1X] == SYMMETRIC ):
            for j in prange(1,NJ-1, nogil=True):
                for k in range(1,NK-1):
                    if self.bc[FACE_1X] == PERIODIC:
                        self.phase[NI-1,j,k] = self.phase[1,j,k]
                    else: # symmetric
                        self.phase[0,j,k] = self.phase[1,j,k]

        # Set phase for the halo lattice sites, boundary face X1
        if ( self.bc[FACE_X1] == PERIODIC or
             self.bc[FACE_X1] == SYMMETRIC ):
            for j in prange(1,NJ-1, nogil=True):
                for k in range(1,NK-1):
                    if self.bc[FACE_X1] == PERIODIC:
                        self.phase[0,j,k] = self.phase[NI-2,j,k]
                    else: # symmetric
                        self.phase[NI-1,j,k] = self.phase[NI-2,j,k]

        # Set phase for the halo lattice sites, boundary face 1Y
        # (note, halo lattice sites in the x-direction included)
        if ( self.bc[FACE_1Y] == PERIODIC or
             self.bc[FACE_1Y] == SYMMETRIC ):
            for i in prange(NI, nogil=True):
                for k in range(1,NK-1):
                    if self.bc[FACE_1Y] == PERIODIC:
                        self.phase[i,NJ-1,k] = self.phase[i,1,k]
                    else: # symmetric
                        self.phase[i,0,k] = self.phase[i,1,k]

        # Set phase for the halo lattice sites, boundary face Y1
        # (note, halo lattice sites in the x-direction included)
        if ( self.bc[FACE_Y1] == PERIODIC or
             self.bc[FACE_Y1] == SYMMETRIC ):
            for i in prange(NI, nogil=True):
                for k in range(1,NK-1):
                    if self.bc[FACE_Y1] == PERIODIC:
                        self.phase[i,0,k] = self.phase[i,NJ-2,k]
                    else: # symmetric
                        self.phase[i,NJ-1,k] = self.phase[i,NJ-2,k]

        # Set phase for the halo lattice sites, boundary face 1Z
        # (note, halo lattice sites in the x- and y-directions included)
        if ( self.bc[FACE_1Z] == PERIODIC or
             self.bc[FACE_1Z] == SYMMETRIC ):
            for i in prange(NI, nogil=True):
                for j in range(NJ):
                    if self.bc[FACE_1Z] == PERIODIC:
                        self.phase[i,j,NK-1] = self.phase[i,j,1]
                    else: # symmetric
                        self.phase[i,j,0] = self.phase[i,j,1]

        # Set phase for the halo lattice sites, boundary face Z1
        # (note, halo lattice sites in the x- and y-directions included)
        if ( self.bc[FACE_Z1] == PERIODIC or
             self.bc[FACE_Z1] == SYMMETRIC ):
            for i in prange(NI, nogil=True):
                for j in range(NJ):
                    if self.bc[FACE_Z1] == PERIODIC:
                        self.phase[i,j,0] = self.phase[i,j,NK-2]
                    else: # symmetric
                        self.phase[i,j,NK-1] = self.phase[i,j,NK-2]

    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cdef void _enforce_den_vel_per_sym_bc(self):
        """Update the density and velocity fields according to the boundary
        conditions, i.e. set density and velocity for the halo lattice sites.
        """
        cdef int i, j, k, NI = self.NI, NJ = self.NJ, NK = self.NK

        # Read density and velocity from the boundary face 1X
        if ( self.bc[FACE_1X] == PERIODIC or
             self.bc[FACE_1X] == SYMMETRIC ):
            for j in prange(1,NJ-1, nogil=True):
                for k in range(1,NK-1):
                    if self.bc[FACE_1X] == PERIODIC:
                        self.den[NI-1,j,k] = self.den[1,j,k]
                        self.vel[NI-1,j,k,X] = self.vel[1,j,k,X]
                        self.vel[NI-1,j,k,Y] = self.vel[1,j,k,Y]
                        self.vel[NI-1,j,k,Z] = self.vel[1,j,k,Z]
                    else: # symmetric
                        self.den[0,j,k] = self.den[1,j,k]
                        self.vel[0,j,k,X] = -self.vel[1,j,k,X] # minus
                        self.vel[0,j,k,Y] = self.vel[1,j,k,Y]
                        self.vel[0,j,k,Z] = self.vel[1,j,k,Z]
                
        # Read density and velocity from the boundary face X1
        if ( self.bc[FACE_X1] == PERIODIC or
             self.bc[FACE_X1] == SYMMETRIC ):
            for j in prange(1,NJ-1, nogil=True):
                for k in range(1,NK-1):
                    if self.bc[FACE_X1] == PERIODIC:
                        self.den[0,j,k] = self.den[NI-2,j,k]
                        self.vel[0,j,k,X] = self.vel[NI-2,j,k,X]
                        self.vel[0,j,k,Y] = self.vel[NI-2,j,k,Y]
                        self.vel[0,j,k,Z] = self.vel[NI-2,j,k,Z]
                    else: # symmetric
                        self.den[NI-1,j,k] = self.den[NI-2,j,k]
                        self.vel[NI-1,j,k,X] = -self.vel[NI-2,j,k,X] # minus
                        self.vel[NI-1,j,k,Y] = self.vel[NI-2,j,k,Y]
                        self.vel[NI-1,j,k,Z] = self.vel[NI-2,j,k,Z]
                
        # Read density and velocity from the boundary face 1Y
        # (note, halo lattice sites in the x-direction included)
        if ( self.bc[FACE_1Y] == PERIODIC or
             self.bc[FACE_1Y] == SYMMETRIC ):
            for i in prange(NI, nogil=True):
                for k in range(1,NK-1):
                    if self.bc[FACE_1Y] == PERIODIC:
                        self.den[i,NJ-1,k] = self.den[i,1,k]
                        self.vel[i,NJ-1,k,X] = self.vel[i,1,k,X]
                        self.vel[i,NJ-1,k,Y] = self.vel[i,1,k,Y]
                        self.vel[i,NJ-1,k,Z] = self.vel[i,1,k,Z]
                    else: # symmetric
                        self.den[i,0,k] = self.den[i,1,k]
                        self.vel[i,0,k,X] = self.vel[i,1,k,X]
                        self.vel[i,0,k,Y] = -self.vel[i,1,k,Y] # minus
                        self.vel[i,0,k,Z] = self.vel[i,1,k,Z]
                
        # Read density and velocity from the boundary face Y1
        # (note, halo lattice sites in the x-direction included)
        if ( self.bc[FACE_Y1] == PERIODIC or
             self.bc[FACE_Y1] == SYMMETRIC ):
            for i in prange(NI, nogil=True):
                for k in range(1,NK-1):
                    if self.bc[FACE_Y1] == PERIODIC:
                        self.den[i,0,k] = self.den[i,NJ-2,k]
                        self.vel[i,0,k,X] = self.vel[i,NJ-2,k,X]
                        self.vel[i,0,k,Y] = self.vel[i,NJ-2,k,Y]
                        self.vel[i,0,k,Z] = self.vel[i,NJ-2,k,Z]
                    else: # symmetric
                        self.den[i,NJ-1,k] = self.den[i,NJ-2,k]
                        self.vel[i,NJ-1,k,X] = self.vel[i,NJ-2,k,X]
                        self.vel[i,NJ-1,k,Y] = -self.vel[i,NJ-2,k,Y] # minus
                        self.vel[i,NJ-1,k,Z] = self.vel[i,NJ-2,k,Z]

        # Read density and velocity from the boundary face 1Z
        # (note, halo lattice sites in the x- and y-directions included)
        if ( self.bc[FACE_1Z] == PERIODIC or
             self.bc[FACE_1Z] == SYMMETRIC ):
            for i in prange(NI, nogil=True):
                for j in range(NJ):
                    if self.bc[FACE_1Z] == PERIODIC:
                        self.den[i,j,NK-1] = self.den[i,j,1]
                        self.vel[i,j,NK-1,X] = self.vel[i,j,1,X]
                        self.vel[i,j,NK-1,Y] = self.vel[i,j,1,Y]
                        self.vel[i,j,NK-1,Z] = self.vel[i,j,1,Z]
                    else: # symmetric
                        self.den[i,j,0] = self.den[i,j,1]
                        self.vel[i,j,0,X] = self.vel[i,j,1,X]
                        self.vel[i,j,0,Y] = self.vel[i,j,1,Y]
                        self.vel[i,j,0,Z] = -self.vel[i,j,1,Z] # minus
                
        # Read density and velocity from the boundary face Z1
        # (note, halo lattice sites in the x- and y-directions included)
        if ( self.bc[FACE_Z1] == PERIODIC or
             self.bc[FACE_Z1] == SYMMETRIC ):
            for i in prange(NI, nogil=True):
                for j in range(NJ):
                    if self.bc[FACE_Z1] == PERIODIC:
                        self.den[i,j,0] = self.den[i,j,NK-2]
                        self.vel[i,j,0,X] = self.vel[i,j,NK-2,X]
                        self.vel[i,j,0,Y] = self.vel[i,j,NK-2,Y]
                        self.vel[i,j,0,Z] = self.vel[i,j,NK-2,Z]
                    else: # symmetric
                        self.den[i,j,NK-1] = self.den[i,j,NK-2]
                        self.vel[i,j,NK-1,X] = self.vel[i,j,NK-2,X]
                        self.vel[i,j,NK-1,Y] = self.vel[i,j,NK-2,Y]
                        self.vel[i,j,NK-1,Z] = -self.vel[i,j,NK-2,Z] # minus
                
    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cdef void _enforce_distr_per_sym_bc(self):
        """Update the distribution fields according to the boundary
        conditions, i.e. set distributions for the halo lattice sites.
        """
        cdef int i, j, k, rn, wn, NI = self.NI, NJ = self.NJ, NK = self.NK

        # Read distributions from the boundary face 1X
        if ( self.bc[FACE_1X] == PERIODIC or
             self.bc[FACE_1X] == SYMMETRIC ):
            for j in prange(1,NJ-1, nogil=True):
                for k in range(1,NK-1):
                    if self.phase[1,j,k] == SOLID:
                        continue
                    rn = self._N_IJK(1,j,k)
                    if self.bc[FACE_1X] == PERIODIC:
                        wn = self._N_IJK(NI-1,j,k)
                        self.f1[self._IND_1I_1J_K0(wn)] = self.f1[self._IND_1I_1J_K0(rn)]
                        self.f1[self._IND_1I_J0_1K(wn)] = self.f1[self._IND_1I_J0_1K(rn)]
                        self.f1[self._IND_1I_J0_K0(wn)] = self.f1[self._IND_1I_J0_K0(rn)]
                        self.f1[self._IND_1I_J0_K1(wn)] = self.f1[self._IND_1I_J0_K1(rn)]
                        self.f1[self._IND_1I_J1_K0(wn)] = self.f1[self._IND_1I_J1_K0(rn)]
                    else: # symmetric
                        wn = self._N_IJK(0,j,k)
                        self.f1[self._IND_I1_1J_K0(wn)] = self.f1[self._IND_1I_1J_K0(rn)]
                        self.f1[self._IND_I1_J0_1K(wn)] = self.f1[self._IND_1I_J0_1K(rn)]
                        self.f1[self._IND_I1_J0_K0(wn)] = self.f1[self._IND_1I_J0_K0(rn)]
                        self.f1[self._IND_I1_J0_K1(wn)] = self.f1[self._IND_1I_J0_K1(rn)]
                        self.f1[self._IND_I1_J1_K0(wn)] = self.f1[self._IND_1I_J1_K0(rn)]
                
        # Read distributions from the boundary face X1
        if ( self.bc[FACE_X1] == PERIODIC or
             self.bc[FACE_X1] == SYMMETRIC ):
            for j in prange(1,NJ-1, nogil=True):
                for k in range(1,NK-1):
                    if self.phase[NI-2,j,k] == SOLID:
                        continue
                    rn = self._N_IJK(NI-2,j,k)
                    if self.bc[FACE_X1] == PERIODIC:
                        wn = self._N_IJK(0,j,k)
                        self.f1[self._IND_I1_1J_K0(wn)] = self.f1[self._IND_I1_1J_K0(rn)]
                        self.f1[self._IND_I1_J0_1K(wn)] = self.f1[self._IND_I1_J0_1K(rn)]
                        self.f1[self._IND_I1_J0_K0(wn)] = self.f1[self._IND_I1_J0_K0(rn)]
                        self.f1[self._IND_I1_J0_K1(wn)] = self.f1[self._IND_I1_J0_K1(rn)]
                        self.f1[self._IND_I1_J1_K0(wn)] = self.f1[self._IND_I1_J1_K0(rn)]
                    else: # symmetric
                        wn = self._N_IJK(NI-1,j,k)
                        self.f1[self._IND_1I_1J_K0(wn)] = self.f1[self._IND_I1_1J_K0(rn)]
                        self.f1[self._IND_1I_J0_1K(wn)] = self.f1[self._IND_I1_J0_1K(rn)]
                        self.f1[self._IND_1I_J0_K0(wn)] = self.f1[self._IND_I1_J0_K0(rn)]
                        self.f1[self._IND_1I_J0_K1(wn)] = self.f1[self._IND_I1_J0_K1(rn)]
                        self.f1[self._IND_1I_J1_K0(wn)] = self.f1[self._IND_I1_J1_K0(rn)]
                
        # Read distributions from the boundary face 1Y
        # (note, halo lattice sites in the x-direction included)
        if ( self.bc[FACE_1Y] == PERIODIC or
             self.bc[FACE_1Y] == SYMMETRIC ):
            for i in prange(NI, nogil=True):
                for k in range(1,NK-1):
                    if self.phase[i,1,k] == SOLID:
                        continue
                    rn = self._N_IJK(i,1,k)
                    if self.bc[FACE_1Y] == PERIODIC:
                        wn = self._N_IJK(i,NJ-1,k)
                        self.f1[self._IND_1I_1J_K0(wn)] = self.f1[self._IND_1I_1J_K0(rn)]
                        self.f1[self._IND_I0_1J_1K(wn)] = self.f1[self._IND_I0_1J_1K(rn)]
                        self.f1[self._IND_I0_1J_K0(wn)] = self.f1[self._IND_I0_1J_K0(rn)]
                        self.f1[self._IND_I0_1J_K1(wn)] = self.f1[self._IND_I0_1J_K1(rn)]
                        self.f1[self._IND_I1_1J_K0(wn)] = self.f1[self._IND_I1_1J_K0(rn)]
                    else: # symmetric
                        wn = self._N_IJK(i,0,k)
                        self.f1[self._IND_1I_J1_K0(wn)] = self.f1[self._IND_1I_1J_K0(rn)]
                        self.f1[self._IND_I0_J1_1K(wn)] = self.f1[self._IND_I0_1J_1K(rn)]
                        self.f1[self._IND_I0_J1_K0(wn)] = self.f1[self._IND_I0_1J_K0(rn)]
                        self.f1[self._IND_I0_J1_K1(wn)] = self.f1[self._IND_I0_1J_K1(rn)]
                        self.f1[self._IND_I1_J1_K0(wn)] = self.f1[self._IND_I1_1J_K0(rn)]
                    
        # Read distributions from the boundary face Y1
        # (note, halo lattice sites in the x-direction included)
        if ( self.bc[FACE_Y1] == PERIODIC or
             self.bc[FACE_Y1] == SYMMETRIC ):
            for i in prange(NI, nogil=True):
                for k in range(1,NK-1):
                    if self.phase[i,NJ-2,k] == SOLID:
                        continue
                    rn = self._N_IJK(i,NJ-2,k)
                    if self.bc[FACE_Y1] == PERIODIC:
                        wn = self._N_IJK(i,0,k)
                        self.f1[self._IND_1I_J1_K0(wn)] = self.f1[self._IND_1I_J1_K0(rn)]
                        self.f1[self._IND_I0_J1_1K(wn)] = self.f1[self._IND_I0_J1_1K(rn)]
                        self.f1[self._IND_I0_J1_K0(wn)] = self.f1[self._IND_I0_J1_K0(rn)]
                        self.f1[self._IND_I0_J1_K1(wn)] = self.f1[self._IND_I0_J1_K1(rn)]
                        self.f1[self._IND_I1_J1_K0(wn)] = self.f1[self._IND_I1_J1_K0(rn)]
                    else: # symmetric
                        wn = self._N_IJK(i,NJ-1,k)
                        self.f1[self._IND_1I_1J_K0(wn)] = self.f1[self._IND_1I_J1_K0(rn)]
                        self.f1[self._IND_I0_1J_1K(wn)] = self.f1[self._IND_I0_J1_1K(rn)]
                        self.f1[self._IND_I0_1J_K0(wn)] = self.f1[self._IND_I0_J1_K0(rn)]
                        self.f1[self._IND_I0_1J_K1(wn)] = self.f1[self._IND_I0_J1_K1(rn)]
                        self.f1[self._IND_I1_1J_K0(wn)] = self.f1[self._IND_I1_J1_K0(rn)]

        # Read distributions from the boundary face 1Z
        # (note, halo lattice sites in the x- and y-directions included)
        if ( self.bc[FACE_1Z] == PERIODIC or
             self.bc[FACE_1Z] == SYMMETRIC ):
            for i in prange(NI, nogil=True):
                for j in range(NJ):
                    if self.phase[i,j,1] == SOLID:
                        continue
                    rn = self._N_IJK(i,j,1)
                    if self.bc[FACE_1Z] == PERIODIC:
                        wn = self._N_IJK(i,j,NK-1)
                        self.f1[self._IND_1I_J0_1K(wn)] = self.f1[self._IND_1I_J0_1K(rn)]
                        self.f1[self._IND_I0_1J_1K(wn)] = self.f1[self._IND_I0_1J_1K(rn)]
                        self.f1[self._IND_I0_J0_1K(wn)] = self.f1[self._IND_I0_J0_1K(rn)]
                        self.f1[self._IND_I0_J1_1K(wn)] = self.f1[self._IND_I0_J1_1K(rn)]
                        self.f1[self._IND_I1_J0_1K(wn)] = self.f1[self._IND_I1_J0_1K(rn)]
                    else: # symmetric
                        wn = self._N_IJK(i,j,0)
                        self.f1[self._IND_1I_J0_K1(wn)] = self.f1[self._IND_1I_J0_1K(rn)]
                        self.f1[self._IND_I0_1J_K1(wn)] = self.f1[self._IND_I0_1J_1K(rn)]
                        self.f1[self._IND_I0_J0_K1(wn)] = self.f1[self._IND_I0_J0_1K(rn)]
                        self.f1[self._IND_I0_J1_K1(wn)] = self.f1[self._IND_I0_J1_1K(rn)]
                        self.f1[self._IND_I1_J0_K1(wn)] = self.f1[self._IND_I1_J0_1K(rn)]
                
        # Read distributions from the boundary face Z1
        # (note, halo lattice sites in the x- and y-directions included)
        if ( self.bc[FACE_Z1] == PERIODIC or
             self.bc[FACE_Z1] == SYMMETRIC ):
            for i in prange(NI, nogil=True):
                for j in range(NJ):
                    if self.phase[i,j,NK-2] == SOLID:
                        continue
                    rn = self._N_IJK(i,j,NK-2)
                    if self.bc[FACE_Z1] == PERIODIC:
                        wn = self._N_IJK(i,j,0)
                        self.f1[self._IND_1I_J0_K1(wn)] = self.f1[self._IND_1I_J0_K1(rn)]
                        self.f1[self._IND_I0_1J_K1(wn)] = self.f1[self._IND_I0_1J_K1(rn)]
                        self.f1[self._IND_I0_J0_K1(wn)] = self.f1[self._IND_I0_J0_K1(rn)]
                        self.f1[self._IND_I0_J1_K1(wn)] = self.f1[self._IND_I0_J1_K1(rn)]
                        self.f1[self._IND_I1_J0_K1(wn)] = self.f1[self._IND_I1_J0_K1(rn)]
                    else: # symmetric
                        wn = self._N_IJK(i,j,NK-1)
                        self.f1[self._IND_1I_J0_1K(wn)] = self.f1[self._IND_1I_J0_K1(rn)]
                        self.f1[self._IND_I0_1J_1K(wn)] = self.f1[self._IND_I0_1J_K1(rn)]
                        self.f1[self._IND_I0_J0_1K(wn)] = self.f1[self._IND_I0_J0_K1(rn)]
                        self.f1[self._IND_I0_J1_1K(wn)] = self.f1[self._IND_I0_J1_K1(rn)]
                        self.f1[self._IND_I1_J0_1K(wn)] = self.f1[self._IND_I1_J0_K1(rn)]
                    
    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cdef void _face_1X_bc(self, int i, int j, int k, double *f) nogil:
        """Enforce fixed variable bc at the domain face 1X.
        """
        cdef:
            double den
            double vel[3]

        # Get boundary condition data
        den = self.bc_data_den_face_1X[j,k]
        vel[X] = self.bc_data_vel_face_1X[j,k,X]
        vel[Y] = self.bc_data_vel_face_1X[j,k,Y]
        vel[Z] = self.bc_data_vel_face_1X[j,k,Z]

        # Assign distributions
        self.bc_func[FACE_1X](den, vel, f)
       
    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cdef void _face_X1_bc(self, int i, int j, int k, double *f) nogil:
        """Enforce fixed variable bc at the domain face X1.
        """
        cdef:
            double den
            double vel[3]

        # Get boundary condition data
        den = self.bc_data_den_face_X1[j,k]
        vel[X] = self.bc_data_vel_face_X1[j,k,X]
        vel[Y] = self.bc_data_vel_face_X1[j,k,Y]
        vel[Z] = self.bc_data_vel_face_X1[j,k,Z]

        # Assign distributions
        self.bc_func[FACE_X1](den, vel, f)

    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cdef void _face_1Y_bc(self, int i, int j, int k, double *f) nogil:
        """Enforce fixed variable bc at the domain face 1Y.
        """
        cdef:
            double den
            double vel[3]

        # Get boundary condition data
        den = self.bc_data_den_face_1Y[i,k]
        vel[X] = self.bc_data_vel_face_1Y[i,k,X]
        vel[Y] = self.bc_data_vel_face_1Y[i,k,Y]
        vel[Z] = self.bc_data_vel_face_1Y[i,k,Z]

        # Assign distributions
        self.bc_func[FACE_1Y](den, vel, f)
       
    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cdef void _face_Y1_bc(self, int i, int j, int k, double *f) nogil:
        """Enforce fixed variable bc at the domain face Y1.
        """
        cdef:
            double den
            double vel[3]

        # Get boundary condition data
        den = self.bc_data_den_face_Y1[i,k]
        vel[X] = self.bc_data_vel_face_Y1[i,k,X]
        vel[Y] = self.bc_data_vel_face_Y1[i,k,Y]
        vel[Z] = self.bc_data_vel_face_Y1[i,k,Z]

        # Assign distributions
        self.bc_func[FACE_Y1](den, vel, f)
       
    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cdef void _face_1Z_bc(self, int i, int j, int k, double *f) nogil:
        """Enforce fixed variable bc at the domain face 1Z.
        """
        cdef:
            double den
            double vel[3]

        # Get boundary condition data
        den = self.bc_data_den_face_1Z[i,j]
        vel[X] = self.bc_data_vel_face_1Z[i,j,X]
        vel[Y] = self.bc_data_vel_face_1Z[i,j,Y]
        vel[Z] = self.bc_data_vel_face_1Z[i,j,Z]

        # Assign distributions
        self.bc_func[FACE_1Z](den, vel, f)
       
    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cdef void _face_Z1_bc(self, int i, int j, int k, double *f) nogil:
        """Enforce fixed variable bc at the domain face Z1.
        """
        cdef:
            double den
            double vel[3]

        # Get boundary condition data
        den = self.bc_data_den_face_Z1[i,j]
        vel[X] = self.bc_data_vel_face_Z1[i,j,X]
        vel[Y] = self.bc_data_vel_face_Z1[i,j,Y]
        vel[Z] = self.bc_data_vel_face_Z1[i,j,Z]

        # Assign distributions
        self.bc_func[FACE_Z1](den, vel, f)
       
    # -----------------------------------------------------------------------
    # Initialization of the distributions (private methods)
    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cdef void _comp_der_vel(self, int i, int j, int k, int fdir,
                            double *vel, double *der_vel) nogil:
        """Compute approximations of local velocity gradients
        (using 2nd-order accurate central finite-differences,
        isotropic in case of non-biased finite-differences).
        """
        cdef:
            int fl, l, li, lj, lk
            int ni, nj, nk, ni_opp, nj_opp, nk_opp
            double ux_nn, uy_nn, uz_nn, ux_nn_opp, uy_nn_opp, uz_nn_opp
            double cff0, cff1, cff2, cff3
        
        der_vel[X] = 0.0
        der_vel[Y] = 0.0
        der_vel[Z] = 0.0
                        
        # Approximate velocity derivatives using central finite-differences
        # (2nd-order accurate, isotropic for non-biased finite-differences)
        for fl in range(self.dvs.FDIR_LVECS):
            l = self.dvs.FDIR_LVEC[fdir][fl]
            li = self.dvs.LI[l]
            lj = self.dvs.LJ[l]
            lk = self.dvs.LK[l]

            ni = i + li
            nj = j + lj
            nk = k + lk

            ni_opp = i - li
            nj_opp = j - lj
            nk_opp = k - lk

            ux_nn = self.vel[ni, nj, nk, X]
            uy_nn = self.vel[ni, nj, nk, Y]
            uz_nn = self.vel[ni, nj, nk, Z]

            ux_nn_opp = self.vel[ni_opp, nj_opp, nk_opp, X]
            uy_nn_opp = self.vel[ni_opp, nj_opp, nk_opp, Y]
            uz_nn_opp = self.vel[ni_opp, nj_opp, nk_opp, Z]

            # non-biased central finite-difference
            cff1 = 0.5
            cff2 = 0.0
            cff3 = -0.5
                            
            if self.phase[ni, nj, nk] == SOLID:
                if self.phase[ni_opp, nj_opp, nk_opp] == SOLID:
                    # non-biased central finite-difference
                    cff1 = 1.0
                    cff3 = -1.0
                else:
                    # forward-biased central finite-difference
                    cff1 = 4.0/3.0
                    cff2 = -1.0
                    cff3 = -1.0/3.0
            elif self.phase[ni_opp, nj_opp, nk_opp] == SOLID:
                # backward-biased central finite-difference
                cff1 = 1.0/3.0
                cff2 = 1.0
                cff3 = -4.0/3.0
                          
            cff0 = 2.0*self.dvs.LW[l]*self.dvs.AS2

            der_vel[X] += cff0*(cff1*ux_nn + cff2*vel[X] + cff3*ux_nn_opp)
            der_vel[Y] += cff0*(cff1*uy_nn + cff2*vel[Y] + cff3*uy_nn_opp)
            der_vel[Z] += cff0*(cff1*uz_nn + cff2*vel[Z] + cff3*uz_nn_opp)

    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cdef void _init_f(self):
        """Initialize distributions.
        """
        cdef:
            int i, j, k, n
            int fn, fluid_lsite_cnt
            double inv_tau = self.rlx_prm[0], tau = 1.0/inv_tau
            double den, m2_neq_cff, rlx_cff = 1.0 - inv_tau
            double *dx_vel
            double *dy_vel
            double *dz_vel
            double *vel
            double *f
            double *h
            
        fluid_lsite_cnt = self.fluid_lsite_domain.shape[1]

        with nogil, parallel():
            f = <double*>malloc(LVECS*sizeof(double))
            h = <double*>malloc(HMOMS*sizeof(double))
            dx_vel = <double*>malloc(3*sizeof(double))
            dy_vel = <double*>malloc(3*sizeof(double))
            dz_vel = <double*>malloc(3*sizeof(double))
            vel = <double*>malloc(3*sizeof(double))

            for fn in prange(fluid_lsite_cnt):
                i = self.fluid_lsite_domain[X,fn] + 1
                j = self.fluid_lsite_domain[Y,fn] + 1
                k = self.fluid_lsite_domain[Z,fn] + 1
                n = self._N_IJK(i,j,k)

                den = self.den[i,j,k]
                vel[X] = self.vel[i,j,k,X]
                vel[Y] = self.vel[i,j,k,Y]
                vel[Z] = self.vel[i,j,k,Z]
                self.dvs.den_vel_to_heq(den, vel, h)
                    
                self._comp_der_vel(i, j, k, FACE_X1, vel, dx_vel)
                self._comp_der_vel(i, j, k, FACE_Y1, vel, dy_vel)
                self._comp_der_vel(i, j, k, FACE_Z1, vel, dz_vel)

                m2_neq_cff = -2.0*tau*self.dvs.INV_AS2*den

                h[H2_XX] += rlx_cff*m2_neq_cff*dx_vel[X]
                h[H2_XY] += rlx_cff*m2_neq_cff*0.5*(dx_vel[Y] + dy_vel[X])
                h[H2_XZ] += rlx_cff*m2_neq_cff*0.5*(dx_vel[Z] + dz_vel[X])
                h[H2_YY] += rlx_cff*m2_neq_cff*dy_vel[Y]
                h[H2_YZ] += rlx_cff*m2_neq_cff*0.5*(dy_vel[Z] + dz_vel[Y])
                h[H2_ZZ] += rlx_cff*m2_neq_cff*dz_vel[Z]
                    
                self.dvs.h_to_f(h, f)
                        
                self.f1[self._IND_1I_1J_K0(n)] = f[L_1I_1J_K0]
                self.f1[self._IND_1I_J0_1K(n)] = f[L_1I_J0_1K]
                self.f1[self._IND_1I_J0_K0(n)] = f[L_1I_J0_K0]
                self.f1[self._IND_1I_J0_K1(n)] = f[L_1I_J0_K1]
                self.f1[self._IND_1I_J1_K0(n)] = f[L_1I_J1_K0]
                self.f1[self._IND_I0_1J_1K(n)] = f[L_I0_1J_1K]
                self.f1[self._IND_I0_1J_K0(n)] = f[L_I0_1J_K0]
                self.f1[self._IND_I0_1J_K1(n)] = f[L_I0_1J_K1]
                self.f1[self._IND_I0_J0_1K(n)] = f[L_I0_J0_1K]
                self.f1[self._IND_I0_J0_K0(n)] = f[L_I0_J0_K0]
                self.f1[self._IND_I0_J0_K1(n)] = f[L_I0_J0_K1]
                self.f1[self._IND_I0_J1_1K(n)] = f[L_I0_J1_1K]
                self.f1[self._IND_I0_J1_K0(n)] = f[L_I0_J1_K0]
                self.f1[self._IND_I0_J1_K1(n)] = f[L_I0_J1_K1]
                self.f1[self._IND_I1_1J_K0(n)] = f[L_I1_1J_K0]
                self.f1[self._IND_I1_J0_1K(n)] = f[L_I1_J0_1K]
                self.f1[self._IND_I1_J0_K0(n)] = f[L_I1_J0_K0]
                self.f1[self._IND_I1_J0_K1(n)] = f[L_I1_J0_K1]
                self.f1[self._IND_I1_J1_K0(n)] = f[L_I1_J1_K0]
                        
            free(dx_vel)
            free(dy_vel)
            free(dz_vel)
            free(vel)
            free(h)
            free(f)
                        
    # -----------------------------------------------------------------------
    # Evolution of the distribution (private methods)
    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cdef void _stream_ijk(self, int i, int j, int k, int n, double *f) nogil:
        """Propagation of the distribution values 
        incorporating halfway-bounceback with moving walls (Ladd's scheme).
        """
        cdef:
            double den = self.den[i,j,k]
            double cff1 = 2.0*den*self.dvs.W1*self.dvs.AS2
            double cff2 = 2.0*den*self.dvs.W2*self.dvs.AS2
        
        # Neighbour 1I_1J_K0
        if self.phase[i-1,j-1,k] != SOLID:
            f[L_I1_J1_K0] = self.f1[self._IND_I1_J1_K0(n+self.njump[L_1I_1J_K0])]
        else:
            f[L_I1_J1_K0] = (self.f1[self._IND_1I_1J_K0(n)] +
                cff2*(self.vel[i-1,j-1,k,X] + self.vel[i-1,j-1,k,Y]))

        # Neighbour 1I_J0_1K
        if self.phase[i-1,j,k-1] != SOLID:
            f[L_I1_J0_K1] = self.f1[self._IND_I1_J0_K1(n+self.njump[L_1I_J0_1K])]
        else:
            f[L_I1_J0_K1] = (self.f1[self._IND_1I_J0_1K(n)] +
                cff2*(self.vel[i-1,j,k-1,X] + self.vel[i-1,j,k-1,Z]))
            
        # Neighbour 1I_J0_K0
        if self.phase[i-1,j,k] != SOLID:
            f[L_I1_J0_K0] = self.f1[self._IND_I1_J0_K0(n+self.njump[L_1I_J0_K0])]
        else:
            f[L_I1_J0_K0] = (self.f1[self._IND_1I_J0_K0(n)] +
                cff1*self.vel[i-1,j,k,X])
            
        # Neighbour 1I_J0_K1
        if self.phase[i-1,j,k+1] != SOLID:
            f[L_I1_J0_1K] = self.f1[self._IND_I1_J0_1K(n+self.njump[L_1I_J0_K1])]
        else:
            f[L_I1_J0_1K] = (self.f1[self._IND_1I_J0_K1(n)] +
                cff2*(self.vel[i-1,j,k+1,X] - self.vel[i-1,j,k+1,Z]))
            
        # Neighbour 1I_J1_K0
        if self.phase[i-1,j+1,k] != SOLID:
            f[L_I1_1J_K0] = self.f1[self._IND_I1_1J_K0(n+self.njump[L_1I_J1_K0])]
        else:
            f[L_I1_1J_K0] = (self.f1[self._IND_1I_J1_K0(n)] +
                cff2*(self.vel[i-1,j+1,k,X] - self.vel[i-1,j+1,k,Y]))
            
        # Neighbour I0_1J_1K
        if self.phase[i,j-1,k-1] != SOLID:
            f[L_I0_J1_K1] = self.f1[self._IND_I0_J1_K1(n+self.njump[L_I0_1J_1K])]
        else:
            f[L_I0_J1_K1] = (self.f1[self._IND_I0_1J_1K(n)] +
                cff2*(self.vel[i,j-1,k-1,Y] + self.vel[i,j-1,k-1,Z]))

        # Neighbour I0_1J_K0
        if self.phase[i,j-1,k] != SOLID:
            f[L_I0_J1_K0] = self.f1[self._IND_I0_J1_K0(n+self.njump[L_I0_1J_K0])]
        else:
            f[L_I0_J1_K0] = (self.f1[self._IND_I0_1J_K0(n)] +
                cff1*self.vel[i,j-1,k,Y])
            
        # Neighbour I0_1J_K1
        if self.phase[i,j-1,k+1] != SOLID:
            f[L_I0_J1_1K] = self.f1[self._IND_I0_J1_1K(n+self.njump[L_I0_1J_K1])]
        else:
            f[L_I0_J1_1K] = (self.f1[self._IND_I0_1J_K1(n)] +
                cff2*(self.vel[i,j-1,k+1,Y] - self.vel[i,j-1,k+1,Z]))
            
        # Neighbour I0_J0_1K
        if self.phase[i,j,k-1] != SOLID:
            f[L_I0_J0_K1] = self.f1[self._IND_I0_J0_K1(n+self.njump[L_I0_J0_1K])]
        else:
            f[L_I0_J0_K1] = (self.f1[self._IND_I0_J0_1K(n)] +
                cff1*self.vel[i,j,k-1,Z])

        # Neighbour I0_J0_K0
        f[L_I0_J0_K0] = self.f1[self._IND_I0_J0_K0(n)]
           
        # Neighbour I0_J0_K1
        if self.phase[i,j,k+1] != SOLID:
            f[L_I0_J0_1K] = self.f1[self._IND_I0_J0_1K(n+self.njump[L_I0_J0_K1])]
        else:
            f[L_I0_J0_1K] = (self.f1[self._IND_I0_J0_K1(n)] -
                cff1*self.vel[i,j,k+1,Z])
            
        # Neighbour I0_J1_1K
        if self.phase[i,j+1,k-1] != SOLID:
            f[L_I0_1J_K1] = self.f1[self._IND_I0_1J_K1(n+self.njump[L_I0_J1_1K])]
        else:
            f[L_I0_1J_K1] = (self.f1[self._IND_I0_J1_1K(n)] -
                cff2*(self.vel[i,j+1,k-1,Y] - self.vel[i,j+1,k-1,Z]))
            
        # Neighbour I0_J1_K0
        if self.phase[i,j+1,k] != SOLID:
            f[L_I0_1J_K0] = self.f1[self._IND_I0_1J_K0(n+self.njump[L_I0_J1_K0])]
        else:
            f[L_I0_1J_K0] = (self.f1[self._IND_I0_J1_K0(n)] -
                cff1*self.vel[i,j+1,k,Y])
            
        # Neighbour I0_J1_K1
        if self.phase[i,j+1,k+1] != SOLID:
            f[L_I0_1J_1K] = self.f1[self._IND_I0_1J_1K(n+self.njump[L_I0_J1_K1])]
        else:
            f[L_I0_1J_1K] = (self.f1[self._IND_I0_J1_K1(n)] -
                cff2*(self.vel[i,j+1,k+1,Y]+self.vel[i,j+1,k+1,Z]))
                        
        # Neighbour I1_1J_K0
        if self.phase[i+1,j-1,k] != SOLID:
            f[L_1I_J1_K0] = self.f1[self._IND_1I_J1_K0(n+self.njump[L_I1_1J_K0])]
        else:
            f[L_1I_J1_K0] = (self.f1[self._IND_I1_1J_K0(n)] -
                cff2*(self.vel[i+1,j-1,k,X] - self.vel[i+1,j-1,k,Y]))
            
        # Neighbour I1_J0_1K
        if self.phase[i+1,j,k-1] != SOLID:
            f[L_1I_J0_K1] = self.f1[self._IND_1I_J0_K1(n+self.njump[L_I1_J0_1K])]
        else:
            f[L_1I_J0_K1] = (self.f1[self._IND_I1_J0_1K(n)] -
                cff2*(self.vel[i+1,j,k-1,X] - self.vel[i+1,j,k-1,Z]))
            
        # Neighbour I1_J0_K0
        if self.phase[i+1,j,k] != SOLID:
            f[L_1I_J0_K0] = self.f1[self._IND_1I_J0_K0(n+self.njump[L_I1_J0_K0])]
        else:
            f[L_1I_J0_K0] = (self.f1[self._IND_I1_J0_K0(n)] -
                cff1*self.vel[i+1,j,k,X])

        # Neighbour I1_J0_K1
        if self.phase[i+1,j,k+1] != SOLID:
            f[L_1I_J0_1K] = self.f1[self._IND_1I_J0_1K(n+self.njump[L_I1_J0_K1])]
        else:
            f[L_1I_J0_1K] = (self.f1[self._IND_I1_J0_K1(n)] -
                cff2*(self.vel[i+1,j,k+1,X] + self.vel[i+1,j,k+1,Z]))

        # Neighbour I1_J1_K0
        if self.phase[i+1,j+1,k] != SOLID:
            f[L_1I_1J_K0] = self.f1[self._IND_1I_1J_K0(n+self.njump[L_I1_J1_K0])]
        else:
            f[L_1I_1J_K0] = (self.f1[self._IND_I1_J1_K0(n)] -
                cff2*(self.vel[i+1,j+1,k,X] + self.vel[i+1,j+1,k,Y]))
            
    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cdef void _update(self):
        """Update distribution values at the fluid lattice sites
        (excluding the halo lattice sites).
        """
        cdef:
            int fn, fluid_lsite_cnt
            int i, j, k, l, n
            double den
            double *vel
            double *acc
            double *f
                     
        fluid_lsite_cnt = self.fluid_lsite_domain.shape[1]
        
        with nogil, parallel():
            f = <double*>malloc(LVECS*sizeof(double))
            vel = <double*>malloc(3*sizeof(double))
            acc = <double*>malloc(3*sizeof(double))

            for fn in prange(fluid_lsite_cnt):
                i = self.fluid_lsite_domain[X,fn] + 1
                j = self.fluid_lsite_domain[Y,fn] + 1
                k = self.fluid_lsite_domain[Z,fn] + 1
                n = self._N_IJK(i,j,k)
                
                self._stream_ijk(i,j,k,n,f)
                
                if i == 1 and self.bc_func[FACE_1X] != NULL:
                    self._face_1X_bc(i, j, k, f)
                if i == self.NI-2 and self.bc_func[FACE_X1] != NULL:
                    self._face_X1_bc(i, j, k, f)
                if j == 1 and self.bc_func[FACE_1Y] != NULL:
                    self._face_1Y_bc(i, j, k, f)
                if j == self.NJ-2 and self.bc_func[FACE_Y1] != NULL:
                    self._face_Y1_bc(i, j, k, f)
                if k == 1 and self.bc_func[FACE_1Z] != NULL:
                    self._face_1Z_bc(i, j, k, f)
                if k == self.NK-2 and self.bc_func[FACE_Z1] != NULL:
                    self._face_Z1_bc(i, j, k, f)

                acc[X] = self.g[X] + self.acc[i,j,k,X]
                acc[Y] = self.g[Y] + self.acc[i,j,k,Y]                      
                acc[Z] = self.g[Z] + self.acc[i,j,k,Z]                      

                den = self.lbe_func(self.dvs, f, acc, self.rlx_prm, vel)

                self.den[i,j,k] = den
                self.vel[i,j,k,X] = vel[X]
                self.vel[i,j,k,Y] = vel[Y]
                self.vel[i,j,k,Z] = vel[Z]

                self.f2[self._IND_1I_1J_K0(n)] = f[L_1I_1J_K0]
                self.f2[self._IND_1I_J0_1K(n)] = f[L_1I_J0_1K]
                self.f2[self._IND_1I_J0_K0(n)] = f[L_1I_J0_K0]
                self.f2[self._IND_1I_J0_K1(n)] = f[L_1I_J0_K1]
                self.f2[self._IND_1I_J1_K0(n)] = f[L_1I_J1_K0]
                self.f2[self._IND_I0_1J_1K(n)] = f[L_I0_1J_1K]
                self.f2[self._IND_I0_1J_K0(n)] = f[L_I0_1J_K0]
                self.f2[self._IND_I0_1J_K1(n)] = f[L_I0_1J_K1]
                self.f2[self._IND_I0_J0_1K(n)] = f[L_I0_J0_1K]
                self.f2[self._IND_I0_J0_K0(n)] = f[L_I0_J0_K0]
                self.f2[self._IND_I0_J0_K1(n)] = f[L_I0_J0_K1]
                self.f2[self._IND_I0_J1_1K(n)] = f[L_I0_J1_1K]
                self.f2[self._IND_I0_J1_K0(n)] = f[L_I0_J1_K0]
                self.f2[self._IND_I0_J1_K1(n)] = f[L_I0_J1_K1]
                self.f2[self._IND_I1_1J_K0(n)] = f[L_I1_1J_K0]
                self.f2[self._IND_I1_J0_1K(n)] = f[L_I1_J0_1K]
                self.f2[self._IND_I1_J0_K0(n)] = f[L_I1_J0_K0]
                self.f2[self._IND_I1_J0_K1(n)] = f[L_I1_J0_K1]
                self.f2[self._IND_I1_J1_K0(n)] = f[L_I1_J1_K0]

            free(f)
            free(vel)
            free(acc)

    # -----------------------------------------------------------------------
    # Update methods (public methods)
    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cpdef unsigned int finalize_init(self):
        """Finalize the solver initialization
        (must be called before the first call to the evolve method).
  
        Raises
        ------
        RuntimeError
           if error has occured in the solver setup

        Returns
        -------
        number of fluid lattice sites in the simul.domain : unsigned int
        """
        cdef:
            int NI = self.NI, NJ = self.NJ, NK = self.NK
            unsigned char [:,:,::1] phaux
            unsigned int fluid_lsite_cnt
        
        if self.error_occured():
            raise RuntimeError('Invalid solver setup: finalization aborted!')
                
        phaux = self.phase[1:NI-1,1:NJ-1,1:NK-1]
        fluid_lsite_cnt = np.count_nonzero(phaux) # Assume SOLID == 0
        
        if fluid_lsite_cnt >= 1:
            self.fluid_lsite_domain = np.array(np.nonzero(phaux),
                                               dtype=np.uint32)
            self._enforce_phase_per_sym_bc()
            self._enforce_den_vel_per_sym_bc()
            self._init_f()

        self.init_done = 1
        return fluid_lsite_cnt
    
    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cpdef unsigned int evolve(self, unsigned int tsteps):
        """Execute the given number of time steps
        (i.e. update distributions values at the fluid lattice sites).
  
        Parameters
        ----------
        tsteps : unsigned int
                 the number of time steps to be executed

        Raises
        ------
        RuntimeError
           if the solver initialization is not finalized
           (i.e. the finalize_init method has not been called)
        RuntimeError
           if error has occured in the solver setup

        Returns
        -------
        the total number of time steps executed (since init.) : unsigned int
        """
        cdef:
            double[:] f_swp
            unsigned int t
            
        if self.error_occured():
            raise RuntimeError('Invalid solver setup: evolve aborted!')
            
        self._set_bc_data() # copy data from den. and vel.fields
         
        if self.init_done == 0:
            mes = 'Finalize solver initialization before calling evolve\n'
            mes += '(i.e. call first the method finalize_init)!'
            raise RuntimeError(mes)
            
        if self.fluid_lsite_domain.shape[1] < 1:
            return self.tot_tsteps
        
        for t in range(tsteps):
            self._enforce_distr_per_sym_bc()
            self._enforce_den_vel_per_sym_bc()
            self._update()
            
            f_swp = self.f1
            self.f1 = self.f2
            self.f2 = f_swp
            
        self.tot_tsteps += tsteps
        return self.tot_tsteps

    # -----------------------------------------------------------------------
    # Status check method (public method)
    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cpdef int error_occured(self):
        return len(self.error_msg)
    
    # -----------------------------------------------------------------------
    # Constitutive equations/material relations (public methods)
    # -----------------------------------------------------------------------
    cpdef double eos_den2p(self, double den):
        """Get pressure as a function of density (equation of state);
        no check for (argument) density > 0.
  
        Parameters
        ----------
        density : double

        Returns
        -------
        pressure : double
        """
        return self.dvs.INV_AS2*den

    # -----------------------------------------------------------------------
    cpdef double eos_p2den(self, double p):
        """Get density as a function of pressure (equation of state);
        no check for (argument) pressure > 0.
  
        Parameters
        ----------
        pressure : double

        Returns
        -------
        density : double
        """
        return self.dvs.AS2*p
    
    # -----------------------------------------------------------------------
    # Getter methods (public methods)
    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cpdef unsigned char[:, :, ::1] get_phase(self):
        """Get the phase field utilized (not updated) by the solver.
  
        Returns
        -------
        phase field : unsigned char[:, :, ::1]
        """
        return self.phase[1:self.NI-1,1:self.NJ-1,1:self.NK-1]
    
    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cpdef double[:, :, :, ::1] get_acc(self):
        """Get the external acceleration field
        utilized (not updated) by the solver.
  
        Returns
        -------
        acc.field : double[:, :, :, ::1]
                    acc[0,0,0,X] = acceler.(x-comp.) for the lat.site (0,0,0)
        """
        return self.acc[1:self.NI-1,1:self.NJ-1,1:self.NK-1,:]

    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cpdef double[:, :, :, ::1] get_vel(self):
        """Get the velocity field updated by the solver.
  
        Returns
        -------
        vel.field : double[:, :, :, ::1]
                    vel[0,0,0,X] = veloc.(x-comp.) for the lat.site (0,0,0)
        """
        return self.vel[1:self.NI-1,1:self.NJ-1,1:self.NK-1,:]

    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cpdef double[:, :, ::1] get_den(self):
        """Get the density field updated by the solver.
  
        Returns
        -------
        density field : double[:, :, ::1]
        """
        return self.den[1:self.NI-1,1:self.NJ-1,1:self.NK-1]
        
    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cpdef unsigned int[:, ::1] get_fluid_lat_sites(self):
        """Get the index coordinates of fluid lattice sites
        in the simulation domain.
  
        Returns
        -------
        sites : unsigned int[:, ::1]
                flsite[X,0] = ind.coord.(i-comp.) of the 1st fluid lat.site
                flsite[Y,0] = ind.coord.(j-comp.) of the 1st fluid lat.site
                flsite[Z,0] = ind.coord.(k-comp.) of the 1st fluid lat.site
        """
        return self.fluid_lsite_domain
        
    # -----------------------------------------------------------------------
    # Indexing and data access functions (private methods)
    # -----------------------------------------------------------------------
    cdef inline int _N_IJK(self, int i, int j, int k) nogil:
      return (i*self.NJ*self.NK + j*self.NK + k)
       
    # -----------------------------------------------------------------------
    cdef inline int _IND_1I_1J_K0(self, int n) nogil:
#      return (L_1I_1J_K0 + n*LVECS) # Collision optimized
#      return (L_1I_1J_K0*self.NCNT + n) # Stream optimized
#      return (n) # Bundle A
#      return (n*5) # Bundle B
      return (n*5) # Bundle C
#      return (n) # Bundle D

    # -----------------------------------------------------------------------
    cdef inline int _IND_1I_J0_1K(self, int n) nogil:
#      return (L_1I_J0_1K + n*LVECS) # Collision optimized
#      return (L_1I_J0_1K*self.NCNT + n) # Stream optimized
#      return (self.NCNT + n*3) # Bundle A
#      return (n*5 + 1) # Bundle B
      return (n*5 + 1) # Bundle C
#      return (self.NCNT + n*3) # Bundle D

    # -----------------------------------------------------------------------
    cdef inline int _IND_1I_J0_K0(self, int n) nogil:
#      return (L_1I_J0_K0 + n*LVECS) # Collision optimized
#      return (L_1I_J0_K0*self.NCNT + n) # Stream optimized
#      return (self.NCNT + n*3 + 1) # Bundle A
#      return (n*5 + 2) # Bundle B
      return (n*5 + 2) # Bundle C
#      return (self.NCNT + n*3 + 1) # Bundle D

    # -----------------------------------------------------------------------
    cdef inline int _IND_1I_J0_K1(self, int n) nogil:
#      return (L_1I_J0_K1 + n*LVECS) # Collision optimized
#      return (L_1I_J0_K1*self.NCNT + n) # Stream optimized
#      return (self.NCNT + n*3 + 2) # Bundle A
#      return (n*5 + 3) # Bundle B
      return (n*5 + 3) # Bundle C
#      return (self.NCNT + n*3 + 2) # Bundle D

    # -----------------------------------------------------------------------
    cdef inline int _IND_1I_J1_K0(self, int n) nogil:
#      return (L_1I_J1_K0 + n*LVECS) # Collision optimized
#      return (L_1I_J1_K0*self.NCNT + n) # Stream optimized
#      return (4*self.NCNT + n) # Bundle A
#      return (n*5 + 4) # Bundle B
      return (n*5 + 4) # Bundle C
#      return (4*self.NCNT + n) # Bundle D

    # -----------------------------------------------------------------------
    cdef inline int _IND_I0_1J_1K(self, int n) nogil:
#      return (L_I0_1J_1K + n*LVECS) # Collision optimized
#      return (L_I0_1J_1K*self.NCNT + n) # Stream optimized
#      return (5*self.NCNT + n*3) # Bundle A
#      return (5*self.NCNT + n*3) # Bundle B
      return (5*self.NCNT + n*9) # Bundle C
#      return (5*self.NCNT + n*9) # Bundle D

    # -----------------------------------------------------------------------
    cdef inline int _IND_I0_1J_K0(self, int n) nogil:
#      return (L_I0_1J_K0 + n*LVECS) # Collision optimized
#      return (L_I0_1J_K0*self.NCNT + n) # Stream optimized
#      return (5*self.NCNT + n*3 + 1) # Bundle A
#      return (5*self.NCNT + n*3 + 1) # Bundle B
      return (5*self.NCNT + n*9 + 1) # Bundle C
#      return (5*self.NCNT + n*9 + 1) # Bundle D

    # -----------------------------------------------------------------------
    cdef inline int _IND_I0_1J_K1(self, int n) nogil:
#      return (L_I0_1J_K1 + n*LVECS) # Collision optimized
#      return (L_I0_1J_K1*self.NCNT + n) # Stream optimized
#      return (5*self.NCNT + n*3 + 2) # Bundle A
#      return (5*self.NCNT + n*3 + 2) # Bundle B
      return (5*self.NCNT + n*9 + 2) # Bundle C
#      return (5*self.NCNT + n*9 + 2) # Bundle D

    # -----------------------------------------------------------------------
    cdef inline int _IND_I0_J0_1K(self, int n) nogil:
#      return (L_I0_J0_1K + n*LVECS) # Collision optimized
#      return (L_I0_J0_1K*self.NCNT + n) # Stream optimized
#      return (8*self.NCNT + n*3) # Bundle A
#      return (8*self.NCNT + n*3) # Bundle B
      return (5*self.NCNT + n*9 + 3) # Bundle C
#      return (5*self.NCNT + n*9 + 3) # Bundle D

    # -----------------------------------------------------------------------
    cdef inline int _IND_I0_J0_K0(self, int n) nogil:
#      return (L_I0_J0_K0 + n*LVECS) # Collision optimized
#      return (L_I0_J0_K0*self.NCNT + n) # Stream optimized
#      return (8*self.NCNT + n*3 + 1) # Bundle A
#      return (8*self.NCNT + n*3 + 1) # Bundle B
      return (5*self.NCNT + n*9 + 4) # Bundle C
#      return (5*self.NCNT + n*9 + 4) # Bundle D

    # -----------------------------------------------------------------------
    cdef inline int _IND_I0_J0_K1(self, int n) nogil:
#      return (L_I0_J0_K1 + n*LVECS) # Collision optimized
#      return (L_I0_J0_K1*self.NCNT + n) # Stream optimized
#      return (8*self.NCNT + n*3 + 2) # Bundle A
#      return (8*self.NCNT + n*3 + 2) # Bundle B
      return (5*self.NCNT + n*9 + 5) # Bundle C
#      return (5*self.NCNT + n*9 + 5) # Bundle D

    # -----------------------------------------------------------------------
    cdef inline int _IND_I0_J1_1K(self, int n) nogil:
#      return (L_I0_J1_1K + n*LVECS) # Collision optimized
#      return (L_I0_J1_1K*self.NCNT + n) # Stream optimized
#      return (11*self.NCNT + n*3) # Bundle A
#      return (11*self.NCNT + n*3) # Bundle B
      return (5*self.NCNT + n*9 + 6) # Bundle C
#      return (5*self.NCNT + n*9 + 6) # Bundle D

    # -----------------------------------------------------------------------
    cdef inline int _IND_I0_J1_K0(self, int n) nogil:
#      return (L_I0_J1_K0 + n*LVECS) # Collision optimized
#      return (L_I0_J1_K0*self.NCNT + n) # Stream optimized
#      return (11*self.NCNT + n*3 + 1) # Bundle A
#      return (11*self.NCNT + n*3 + 1) # Bundle B
      return (5*self.NCNT + n*9 + 7) # Bundle C
#      return (5*self.NCNT + n*9 + 7) # Bundle D

    # -----------------------------------------------------------------------
    cdef inline int _IND_I0_J1_K1(self, int n) nogil:
#      return (L_I0_J1_K1 + n*LVECS) # Collision optimized
#      return (L_I0_J1_K1TN*self.NCNT + n) # Stream optimized
#      return (11*self.NCNT + n*3 + 2) # Bundle A
#      return (11*self.NCNT + n*3 + 2) # Bundle B
      return (5*self.NCNT + n*9 + 8) # Bundle C
#      return (5*self.NCNT + n*9 + 8) # Bundle D

    # -----------------------------------------------------------------------
    cdef inline int _IND_I1_1J_K0(self, int n) nogil:
#      return (L_I1_1J_K0 + n*LVECS) # Collision optimized
#      return (L_I1_1J_K0*self.NCNT + n) # Stream optimized
#      return (14*self.NCNT + n) # Bundle A
#      return (14*self.NCNT + n*5) # Bundle B
      return (14*self.NCNT + n*5) # Bundle C
#      return (14*self.NCNT + n) # Bundle D

    # -----------------------------------------------------------------------
    cdef inline int _IND_I1_J0_1K(self, int n) nogil:
#      return (L_I1_J0_1K + n*LVECS) # Collision optimized
#      return (L_I1_J0_1K*self.NCNT + n) # Stream optimized
#      return (15*self.NCNT + n*3) # Bundle A
#      return (14*self.NCNT + n*5 + 1) # Bundle B
      return (14*self.NCNT + n*5 + 1) # Bundle C
#      return (15*self.NCNT + n*3) # Bundle D

    # -----------------------------------------------------------------------
    cdef inline int _IND_I1_J0_K0(self, int n) nogil:
#      return (L_I1_J0_K0 + n*LVECS) # Collision optimized
#      return (L_I1_J0_K0*self.NCNT + n) # Stream optimized
#      return (15*self.NCNT + n*3 + 1) # Bundle A
#      return (14*self.NCNT + n*5 + 2) # Bundle B
      return (14*self.NCNT + n*5 + 2) # Bundle C
#      return (15*self.NCNT + n*3 + 1) # Bundle D

    # -----------------------------------------------------------------------
    cdef inline int _IND_I1_J0_K1(self, int n) nogil:
#      return (L_I1_J0_K1 + n*LVECS) # Collision optimized
#      return (L_I1_J0_K1*self.NCNT + n) # Stream optimized
#      return (15*self.NCNT + n*3 + 2) # Bundle A
#      return (14*self.NCNT + n*5 + 3) # Bundle B
      return (14*self.NCNT + n*5 + 3) # Bundle C
#      return (15*self.NCNT + n*3 + 2) # Bundle D

    # -----------------------------------------------------------------------
    cdef inline int _IND_I1_J1_K0(self, int n) nogil:
#      return (L_I1_J1_K0 + n*LVECS) # Collision optimized
#      return (L_I1_J1_K0*self.NCNT + n) # Stream optimized
#      return (18*self.NCNT + n) # Bundle A
#      return (14*self.NCNT + n*5 + 4) # Bundle B
      return (14*self.NCNT + n*5 + 4) # Bundle C
#      return (18*self.NCNT + n) # Bundle D

    # -----------------------------------------------------------------------
