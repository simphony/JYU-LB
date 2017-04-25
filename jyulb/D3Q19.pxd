"""Specification of the D3Q19 discrete velocity set,
an associated Hermite moment space representation (H10),
transformations between distributions and moments,
and implementation of the forward-euler updata procedure (standard LBE)
with well-known relaxation schemes.

Details
-------
A lattice vector is defined by a lattice index vector (i,j,k), i.e.

LX(i,j,k) = i*B1X + j*B2X + k*B3X,
LY(i,j,k) = i*B1Y + j*B2Y + k*B3Y,   
LZ(i,j,k) = i*B1Z + j*B2Z + k*B3Z,    

where B1, B2, and B3 are base vectors of the lattice. Note the relation
between dimensionless velocity vectors and the lattice vectors, i.e.

VX(i,j,k) = AS*LX(i,j,k),
VY(i,j,k) = AS*LY(i,j,k),
VZ(i,j,k) = AS*LZ(i,j,k),

where AS is the so-called scaling factor of a discrete velocity set.

Finally, the microscopic velocity vectors are defined either as

CX(i,j,k) = CR*LX(i,j,k),
CY(i,j,k) = CR*LY(i,j,k),
CZ(i,j,k) = CR*LZ(i,j,k),

or as 

CX(i,j,k) = CT*VX(i,j,k),
CY(i,j,k) = CT*VY(i,j,k),
CZ(i,j,k) = CT*VZ(i,j,k);

CR is the so-called lattice speed (CR = DR/DT, DR is the lattice
spacing while DT is the discrete time step) and CT the thermal speed
(CT = sqrt[Kb T/m], Kb is the Boltzmann constant (J/K), T is the temperature
while m is the molecular mass). The above implies relation CT = CR/AS.

Important: here all the moments are computed and treated
           as sums over the lattice vectors!

Author
------
Keijo Mattila, JYU, April 2017.
"""
# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
import cython
cimport cython

# ---------------------------------------------------------------------------
# Enumeration of lattice index vectors for D3Q19
# Notation: L_1I_1J_K0 denotes the lattice index vector = (-1,-1,0),
#           L_I1_J0_K1 denotes the lattice index vector = (1,0,1), etc.
# ---------------------------------------------------------------------------
cdef enum:
    L_1I_1J_K0 = 0
    L_1I_J0_1K = 1
    L_1I_J0_K0 = 2
    L_1I_J0_K1 = 3
    L_1I_J1_K0 = 4
    L_I0_1J_1K = 5
    L_I0_1J_K0 = 6
    L_I0_1J_K1 = 7
    L_I0_J0_1K = 8
    L_I0_J0_K0 = 9
    L_I0_J0_K1 = 10
    L_I0_J1_1K = 11
    L_I0_J1_K0 = 12
    L_I0_J1_K1 = 13
    L_I1_1J_K0 = 14
    L_I1_J0_1K = 15
    L_I1_J0_K0 = 16
    L_I1_J0_K1 = 17
    L_I1_J1_K0 = 18
    LVECS      = 19

# ---------------------------------------------------------------------------
# Enumeration of Hermite polynomial moments
# for the H10 moment space representation
# (involves 10 Hermite polynomial moments up to rank 2)
# ---------------------------------------------------------------------------
cdef enum:
    H0 = 0     # density
    H1_X = 1   # momentum density, x-component
    H1_Y = 2   # momentum density, y-component
    H1_Z = 3   # momentum density, z-component
    H2_XX = 4 
    H2_XY = 5
    H2_XZ = 6
    H2_YY = 7
    H2_YZ = 8
    H2_ZZ = 9
    HMOMS = 10
    
# ---------------------------------------------------------------------------
# Data type definitions (function pointer to an update procedure)
# ---------------------------------------------------------------------------
ctypedef double (*lbe_fpntr)(D3Q19_H10,double*,double*,double*,double*) nogil
    
# ---------------------------------------------------------------------------
# Class D3Q19_H10:
#     a data structure which specifies the D3Q19 discrete velocity set
#     together with the H10 moment space representation
# ---------------------------------------------------------------------------
cdef class D3Q19_H10:
    # Base vectors of a cubic lattice
     # (standard basis of the cartesian coordinate system)
    cdef readonly double B1[3] # = (1,0,0)
    cdef readonly double B2[3] # = (0,1,0)
    cdef readonly double B3[3] # = (0,0,1)

    # Scaling factor
    cdef readonly double AS, AS2, AS4, INV_AS, INV_AS2, INV_AS4

    # Weight coefficients
    cdef readonly double W0, W1, W2

    # Lattice index vectors (i-, j-, and k-components)
    cdef readonly int[LVECS] LI
    cdef readonly int[LVECS] LJ
    cdef readonly int[LVECS] LK

    # Lattice vectors (x-, y-, and z-components)
    cdef readonly double[LVECS] LX
    cdef readonly double[LVECS] LY
    cdef readonly double[LVECS] LZ

    # Opposite lattice vectors
    # (denoted using the index vector enumeration numbers)
    cdef readonly unsigned char[LVECS] OPP_LVEC

    # Weight coefficients per lattice vector
    cdef readonly double[LVECS] LW

    # Lattice vectors in the direction of cuboid face normals
    # (a cuboid aligned with the axes of cartesian coordinate system)
    cdef readonly int FDIR_LVECS # number of vectors = 5
    cdef readonly int FDIR_LVEC[6][5]

    # Range and span of lattice index vectors
    cdef readonly int LRANGE # i.e. max.value of a latt.index vect.component
    cdef readonly int LSPAN # = 2*LRANGE + 1
    
    # Hermite polynomials (order 2)
    cdef readonly double HPOL2_XX[LVECS]
    cdef readonly double HPOL2_XY[LVECS]
    cdef readonly double HPOL2_XZ[LVECS]
    cdef readonly double HPOL2_YY[LVECS]
    cdef readonly double HPOL2_YZ[LVECS]
    cdef readonly double HPOL2_ZZ[LVECS]

    # Kinetic projectors for Hermite polynomial moments up to rank 2
    cdef readonly double K1_X[LVECS]
    cdef readonly double K1_Y[LVECS]
    cdef readonly double K1_Z[LVECS]
    cdef readonly double K2_XX[LVECS]
    cdef readonly double K2_XY[LVECS]
    cdef readonly double K2_XZ[LVECS]
    cdef readonly double K2_YY[LVECS]
    cdef readonly double K2_YZ[LVECS]
    cdef readonly double K2_ZZ[LVECS]
    
    # -----------------------------------------------------------------------
    # Functions for computing moments of the distributions
    # (i.e. transformation of distributions into moments)
    # -----------------------------------------------------------------------
    # Note: the discr.moments are computed as sums over the lattice vectors
    #       (not as sums over the dimensionless velocity vectors)
    # -----------------------------------------------------------------------
    cdef double f_to_den(self, double *f) nogil
    cdef double f_to_den_mom(self, double *f, double *mom) nogil
    cdef double f_to_den_vel(self, double *f, double *vel) nogil
    cdef void f_to_h(self, double *f, double *h) nogil

    # -----------------------------------------------------------------------
    # Functions for projecting moments into the kinetic space
    # (i.e. transformation of moments into distributions)
    # -----------------------------------------------------------------------
    # Assumption: the moments are treated as sums over the lattice vectors
    #             (not as sums over the dimensionless velocity vectors)
    # -----------------------------------------------------------------------
    cdef void h_to_f(self, double *h, double *f) nogil
    cdef void add_h_to_f(self, double *h, double *f) nogil
    cdef void den_vel_to_heq(self,double den,double *vel,double *heq) nogil
    cdef void eq(self, double den, double *vel, double *feq) nogil

    # -----------------------------------------------------------------------
    # Forward-euler updata procedures (standard LBE)
    # with well-known relaxation schemes
    # -----------------------------------------------------------------------
    cdef double forw_euler_BGK(self, double *f, double *acc, double *prm,
                               double *vel) nogil
    cdef double forw_euler_TRT(self, double *f, double *acc, double *prm,
                               double *vel) nogil
    cdef double forw_euler_regul(self, double *f, double *acc, double *prm,
                                 double *vel) nogil
    # -----------------------------------------------------------------------
    