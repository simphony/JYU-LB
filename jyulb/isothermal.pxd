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

from bcs cimport bcs_fpntr
from D3Q19 cimport lbe_fpntr
from D3Q19 cimport D3Q19_H10, LVECS

# ---------------------------------------------------------------------------
# Class Solver
# ---------------------------------------------------------------------------
cdef class Solver:
    # ------------------
    # Private attributes
    # ------------------
    cdef readonly int NI, NJ, NK, NCNT
    cdef readonly unsigned tot_tsteps
    cdef readonly int init_done
    
    cdef readonly int bc[6]
    cdef readonly double g[3]
    cdef readonly double kvisc

    cdef readonly int rlx_scheme
    cdef readonly double rlx_prm[LVECS]

    cdef lbe_fpntr lbe_func
    cdef bcs_fpntr bc_func[6]
    
    cdef readonly unsigned char[:, :, ::1] phase
    cdef readonly double[:, :, :, ::1] acc
    cdef readonly double[:, :, :, ::1] vel
    cdef readonly double[:, :, ::1] den

    cdef readonly D3Q19_H10 dvs
    cdef readonly double[:] f1
    cdef readonly double[:] f2
    
    cdef readonly double[:, ::1] bc_data_den_face_1X
    cdef readonly double[:, ::1] bc_data_den_face_X1
    cdef readonly double[:, ::1] bc_data_den_face_1Y
    cdef readonly double[:, ::1] bc_data_den_face_Y1
    cdef readonly double[:, ::1] bc_data_den_face_1Z
    cdef readonly double[:, ::1] bc_data_den_face_Z1

    cdef readonly double[:, :, ::1] bc_data_vel_face_1X
    cdef readonly double[:, :, ::1] bc_data_vel_face_X1
    cdef readonly double[:, :, ::1] bc_data_vel_face_1Y
    cdef readonly double[:, :, ::1] bc_data_vel_face_Y1
    cdef readonly double[:, :, ::1] bc_data_vel_face_1Z
    cdef readonly double[:, :, ::1] bc_data_vel_face_Z1
    
    cdef readonly int njump[LVECS]
    cdef readonly unsigned int[:, ::1] fluid_lsite_domain

    cdef readonly str error_msg
    
    # ---------------------------------------------------------------------
    # Public methods (i.e. the python interface excluding the constructors)
    # ---------------------------------------------------------------------
    # Update methods
    cpdef unsigned int finalize_init(self)
    cpdef unsigned int evolve(self, unsigned int tsteps)

    # Status check method
    cpdef int error_occured(self)
    
    # Getter methods
    cpdef unsigned char[:, :, ::1] get_phase(self)
    cpdef double[:, :, :, ::1] get_acc(self)
    cpdef double[:, :, :, ::1] get_vel(self)
    cpdef double[:, :, ::1] get_den(self)
    cpdef unsigned int[:, ::1] get_fluid_lat_sites(self)
    
    # Constitutive equations/material relations 
    cpdef double eos_den2p(self, double den)
    cpdef double eos_p2den(self, double p)
    
    # ---------------
    # Private methods
    # ---------------
    # Configuration of the boundary conditions
    cdef void _check_bc(self, int[:] bc)
    cdef void _init_bc_scheme(self)
    cdef void _set_bc_data(self)

    # Initialization of the distributions
    cdef void _comp_der_vel(self, int i, int j, int k, int fdir,
                            double *vel, double *der_vel) nogil
    cdef void _init_f(self)
    
    # Enforcement of the boundary conditions
    cdef void _enforce_phase_per_sym_bc(self)
    cdef void _enforce_den_vel_per_sym_bc(self)
    cdef void _enforce_distr_per_sym_bc(self)

    cdef void _face_1X_bc(self, int i, int j, int k, double *f) nogil
    cdef void _face_X1_bc(self, int i, int j, int k, double *f) nogil
    cdef void _face_1Y_bc(self, int i, int j, int k, double *f) nogil
    cdef void _face_Y1_bc(self, int i, int j, int k, double *f) nogil
    cdef void _face_1Z_bc(self, int i, int j, int k, double *f) nogil
    cdef void _face_Z1_bc(self, int i, int j, int k, double *f) nogil
                                   
    # Evolution of the distributions
    cdef void _stream_ijk(self, int i, int j, int k, int n, double *f) nogil
    cdef void _update(self)

    # Indexing and data access functions
    cdef int _N_IJK(self, int i, int j, int k) nogil

    cdef int _IND_1I_1J_K0(self, int n) nogil
    cdef int _IND_1I_J0_1K(self, int n) nogil
    cdef int _IND_1I_J0_K0(self, int n) nogil
    cdef int _IND_1I_J0_K1(self, int n) nogil
    cdef int _IND_1I_J1_K0(self, int n) nogil
    cdef int _IND_I0_1J_1K(self, int n) nogil
    cdef int _IND_I0_1J_K0(self, int n) nogil
    cdef int _IND_I0_1J_K1(self, int n) nogil
    cdef int _IND_I0_J0_1K(self, int n) nogil
    cdef int _IND_I0_J0_K0(self, int n) nogil
    cdef int _IND_I0_J0_K1(self, int n) nogil
    cdef int _IND_I0_J1_1K(self, int n) nogil
    cdef int _IND_I0_J1_K0(self, int n) nogil
    cdef int _IND_I0_J1_K1(self, int n) nogil
    cdef int _IND_I1_1J_K0(self, int n) nogil
    cdef int _IND_I1_J0_1K(self, int n) nogil
    cdef int _IND_I1_J0_K0(self, int n) nogil
    cdef int _IND_I1_J0_K1(self, int n) nogil
    cdef int _IND_I1_J1_K0(self, int n) nogil
    
# ---------------------------------------------------------------------------