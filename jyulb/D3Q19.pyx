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
import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound, cdivision

from defs cimport FACE_1X,FACE_X1,FACE_1Y,FACE_Y1,FACE_1Z,FACE_Z1
from defs cimport BGK, TRT, REGULARIZATION, RECURRENCE_REL
from defs cimport X,Y,Z

# ---------------------------------------------------------------------------
# Implementation of class D3Q19_H10
# ---------------------------------------------------------------------------
cdef class D3Q19_H10:

    def __cinit__(self):
        cdef:
            int l
            double AS = np.sqrt(3.0), AS2 = 3.0, AS4 = AS2*AS2
            double INV_AS = 1.0/AS, INV_AS2 = 1.0/AS2, INV_AS4 = 1.0/AS4
            double W0 = 12.0/36.0, W1 =  2.0/36.0, W2 =  1.0/36.0
            double B1X = 1.0, B1Y = 0.0, B1Z = 0.0
            double B2X = 0.0, B2Y = 1.0, B2Z = 0.0
            double B3X = 0.0, B3Y = 0.0, B3Z = 1.0

        # Base vectors of a cubic lattice
        self.B1[X] = B1X
        self.B1[Y] = B1Y
        self.B1[Z] = B1Z

        self.B2[X] = B2X
        self.B2[Y] = B2Y
        self.B2[Z] = B2Z

        self.B3[X] = B3X
        self.B3[Y] = B3Y
        self.B3[Z] = B3Z

        # Scaling factor
        self.AS = AS
        self.AS2 = AS2
        self.AS4 = AS4
        self.INV_AS = INV_AS
        self.INV_AS2 = INV_AS2
        self.INV_AS4 = INV_AS4

        # Weight coefficients
        self.W0 = W0
        self.W1 = W1
        self.W2 = W2

        # Lattice index vectors, i-components
        self.LI[L_1I_1J_K0] = -1
        self.LI[L_1I_J0_1K] = -1
        self.LI[L_1I_J0_K0] = -1
        self.LI[L_1I_J0_K1] = -1
        self.LI[L_1I_J1_K0] = -1
        self.LI[L_I0_1J_1K] = 0
        self.LI[L_I0_1J_K0] = 0
        self.LI[L_I0_1J_K1] = 0
        self.LI[L_I0_J0_1K] = 0
        self.LI[L_I0_J0_K0] = 0
        self.LI[L_I0_J0_K1] = 0
        self.LI[L_I0_J1_1K] = 0
        self.LI[L_I0_J1_K0] = 0
        self.LI[L_I0_J1_K1] = 0
        self.LI[L_I1_1J_K0] = 1
        self.LI[L_I1_J0_1K] = 1
        self.LI[L_I1_J0_K0] = 1
        self.LI[L_I1_J0_K1] = 1
        self.LI[L_I1_J1_K0] = 1
        
        # Lattice index vectors, j-components
        self.LJ[L_1I_1J_K0] = -1
        self.LJ[L_1I_J0_1K] = 0
        self.LJ[L_1I_J0_K0] = 0
        self.LJ[L_1I_J0_K1] = 0
        self.LJ[L_1I_J1_K0] = 1
        self.LJ[L_I0_1J_1K] = -1
        self.LJ[L_I0_1J_K0] = -1
        self.LJ[L_I0_1J_K1] = -1
        self.LJ[L_I0_J0_1K] = 0
        self.LJ[L_I0_J0_K0] = 0
        self.LJ[L_I0_J0_K1] = 0
        self.LJ[L_I0_J1_1K] = 1
        self.LJ[L_I0_J1_K0] = 1
        self.LJ[L_I0_J1_K1] = 1
        self.LJ[L_I1_1J_K0] = -1
        self.LJ[L_I1_J0_1K] = 0
        self.LJ[L_I1_J0_K0] = 0
        self.LJ[L_I1_J0_K1] = 0
        self.LJ[L_I1_J1_K0] = 1

        # Lattice index vectors, k-components
        self.LK[L_1I_1J_K0] = 0
        self.LK[L_1I_J0_1K] = -1
        self.LK[L_1I_J0_K0] = 0
        self.LK[L_1I_J0_K1] = 1
        self.LK[L_1I_J1_K0] = 0
        self.LK[L_I0_1J_1K] = -1
        self.LK[L_I0_1J_K0] = 0
        self.LK[L_I0_1J_K1] = 1
        self.LK[L_I0_J0_1K] = -1
        self.LK[L_I0_J0_K0] = 0
        self.LK[L_I0_J0_K1] = 1
        self.LK[L_I0_J1_1K] = -1
        self.LK[L_I0_J1_K0] = 0
        self.LK[L_I0_J1_K1] = 1
        self.LK[L_I1_1J_K0] = 0
        self.LK[L_I1_J0_1K] = -1
        self.LK[L_I1_J0_K0] = 0
        self.LK[L_I1_J0_K1] = 1
        self.LK[L_I1_J1_K0] = 0

        # Lattice vectors
        for l in range(LVECS):
            self.LX[l] = B1X*self.LI[l] + B2X*self.LJ[l] + B3X*self.LK[l]
            self.LY[l] = B1Y*self.LI[l] + B2Y*self.LJ[l] + B3Y*self.LK[l]
            self.LZ[l] = B1Z*self.LI[l] + B2Z*self.LJ[l] + B3Z*self.LK[l]
       
        # Opposite lattice vectors, enumeration number
        self.OPP_LVEC[L_1I_1J_K0] = L_I1_J1_K0
        self.OPP_LVEC[L_1I_J0_1K] = L_I1_J0_K1
        self.OPP_LVEC[L_1I_J0_K0] = L_I1_J0_K0
        self.OPP_LVEC[L_1I_J0_K1] = L_I1_J0_1K
        self.OPP_LVEC[L_1I_J1_K0] = L_I1_1J_K0
        self.OPP_LVEC[L_I0_1J_1K] = L_I0_J1_K1
        self.OPP_LVEC[L_I0_1J_K0] = L_I0_J1_K0
        self.OPP_LVEC[L_I0_1J_K1] = L_I0_J1_1K
        self.OPP_LVEC[L_I0_J0_1K] = L_I0_J0_K1
        self.OPP_LVEC[L_I0_J0_K0] = L_I0_J0_K0
        self.OPP_LVEC[L_I0_J0_K1] = L_I0_J0_1K
        self.OPP_LVEC[L_I0_J1_1K] = L_I0_1J_K1
        self.OPP_LVEC[L_I0_J1_K0] = L_I0_1J_K0
        self.OPP_LVEC[L_I0_J1_K1] = L_I0_1J_1K
        self.OPP_LVEC[L_I1_1J_K0] = L_1I_J1_K0
        self.OPP_LVEC[L_I1_J0_1K] = L_1I_J0_K1
        self.OPP_LVEC[L_I1_J0_K0] = L_1I_J0_K0
        self.OPP_LVEC[L_I1_J0_K1] = L_1I_J0_1K
        self.OPP_LVEC[L_I1_J1_K0] = L_1I_1J_K0

        # Weight coefficients per lattice vector
        self.LW[L_1I_1J_K0] = W2
        self.LW[L_1I_J0_1K] = W2
        self.LW[L_1I_J0_K0] = W1
        self.LW[L_1I_J0_K1] = W2
        self.LW[L_1I_J1_K0] = W2
        self.LW[L_I0_1J_1K] = W2
        self.LW[L_I0_1J_K0] = W1
        self.LW[L_I0_1J_K1] = W2
        self.LW[L_I0_J0_1K] = W1
        self.LW[L_I0_J0_K0] = W0
        self.LW[L_I0_J0_K1] = W1
        self.LW[L_I0_J1_1K] = W2
        self.LW[L_I0_J1_K0] = W1
        self.LW[L_I0_J1_K1] = W2
        self.LW[L_I1_1J_K0] = W2
        self.LW[L_I1_J0_1K] = W2
        self.LW[L_I1_J0_K0] = W1
        self.LW[L_I1_J0_K1] = W2
        self.LW[L_I1_J1_K0] = W2
        
        # Lattice vectors in the direction of cuboid face normals
        # (a cuboid aligned with the axes of cartesian coordinate system)
        self.FDIR_LVECS = 5
        
        self.FDIR_LVEC[FACE_1X][0] = L_1I_1J_K0
        self.FDIR_LVEC[FACE_1X][1] = L_1I_J0_1K
        self.FDIR_LVEC[FACE_1X][2] = L_1I_J0_K0
        self.FDIR_LVEC[FACE_1X][3] = L_1I_J0_K1
        self.FDIR_LVEC[FACE_1X][4] = L_1I_J1_K0

        self.FDIR_LVEC[FACE_X1][0] = L_I1_1J_K0
        self.FDIR_LVEC[FACE_X1][1] = L_I1_J0_1K
        self.FDIR_LVEC[FACE_X1][2] = L_I1_J0_K0
        self.FDIR_LVEC[FACE_X1][3] = L_I1_J0_K1
        self.FDIR_LVEC[FACE_X1][4] = L_I1_J1_K0

        self.FDIR_LVEC[FACE_1Y][0] = L_1I_1J_K0
        self.FDIR_LVEC[FACE_1Y][1] = L_I0_1J_1K
        self.FDIR_LVEC[FACE_1Y][2] = L_I0_1J_K0
        self.FDIR_LVEC[FACE_1Y][3] = L_I0_1J_K1
        self.FDIR_LVEC[FACE_1Y][4] = L_I1_1J_K0

        self.FDIR_LVEC[FACE_Y1][0] = L_1I_J1_K0
        self.FDIR_LVEC[FACE_Y1][1] = L_I0_J1_1K
        self.FDIR_LVEC[FACE_Y1][2] = L_I0_J1_K0
        self.FDIR_LVEC[FACE_Y1][3] = L_I0_J1_K1
        self.FDIR_LVEC[FACE_Y1][4] = L_I1_J1_K0

        self.FDIR_LVEC[FACE_1Z][0] = L_1I_J0_1K
        self.FDIR_LVEC[FACE_1Z][1] = L_I0_1J_1K
        self.FDIR_LVEC[FACE_1Z][2] = L_I0_J0_1K
        self.FDIR_LVEC[FACE_1Z][3] = L_I0_J1_1K
        self.FDIR_LVEC[FACE_1Z][4] = L_I1_J0_1K

        self.FDIR_LVEC[FACE_Z1][0] = L_1I_J0_K1
        self.FDIR_LVEC[FACE_Z1][1] = L_I0_1J_K1
        self.FDIR_LVEC[FACE_Z1][2] = L_I0_J0_K1
        self.FDIR_LVEC[FACE_Z1][3] = L_I0_J1_K1
        self.FDIR_LVEC[FACE_Z1][4] = L_I1_J0_K1
        
        # Range and span of lattice index vectors
        self.LRANGE = 1
        self.LSPAN = 2*self.LRANGE + 1

        # Hermite polynomials and kinetic projectors
        for l in range(LVECS):
            self.HPOL2_XX[l] = self.LX[l]*self.LX[l] - self.INV_AS2
            self.HPOL2_XY[l] = self.LX[l]*self.LY[l]
            self.HPOL2_XZ[l] = self.LX[l]*self.LZ[l]
            self.HPOL2_YY[l] = self.LY[l]*self.LY[l] - self.INV_AS2
            self.HPOL2_YZ[l] = self.LY[l]*self.LZ[l]
            self.HPOL2_ZZ[l] = self.LZ[l]*self.LZ[l] - self.INV_AS2
            
            self.K1_X[l] = self.AS2*self.LW[l]*self.LX[l]
            self.K1_Y[l] = self.AS2*self.LW[l]*self.LY[l]
            self.K1_Z[l] = self.AS2*self.LW[l]*self.LZ[l]

            self.K2_XX[l] = 0.5*self.AS4*self.LW[l]*self.HPOL2_XX[l]
            self.K2_XY[l] = 0.5*self.AS4*self.LW[l]*self.HPOL2_XY[l]
            self.K2_XZ[l] = 0.5*self.AS4*self.LW[l]*self.HPOL2_XZ[l]
            self.K2_YY[l] = 0.5*self.AS4*self.LW[l]*self.HPOL2_YY[l]
            self.K2_YZ[l] = 0.5*self.AS4*self.LW[l]*self.HPOL2_YZ[l]
            self.K2_ZZ[l] = 0.5*self.AS4*self.LW[l]*self.HPOL2_ZZ[l]
        
    # -----------------------------------------------------------------------
    # Functions for computing moments of the distributions
    # (i.e. transformation of distributions into moments)
    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cdef inline double f_to_den(self, double *f) nogil:
        """Compute (mass) density from the distribution values.
        
        Parameters
        ----------
        f : double* (array, element count >= 19, read)
            distribution values

        Returns
        -------
        density : double
        """
        return (f[L_1I_1J_K0] + f[L_1I_J0_1K] + f[L_1I_J0_K0] +
                f[L_1I_J0_K1] + f[L_1I_J1_K0] + f[L_I0_1J_1K] +
                f[L_I0_1J_K0] + f[L_I0_1J_K0] + f[L_I0_J0_1K] +
                f[L_I0_J0_K0] + f[L_I0_J0_K0] + f[L_I0_J1_1K] +
                f[L_I0_J1_K0] + f[L_I0_J1_K0] + f[L_I1_1J_K0] +
                f[L_I1_J0_1K] + f[L_I1_J0_K0] + f[L_I1_J0_K1] +
                f[L_I1_J1_K0])
                    
    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cdef inline double f_to_den_mom(self, double *f, double *mom) nogil:
        """Compute mass and momentum density from the distribution values.
        
        Parameters
        ----------
        f : double* (array, element count >= 19, read)
            distribution values
        mom : double* (array, element count >= 3, write)
              momentum density vector

        Returns
        -------
        density : double
        """
        cdef:
            double den_1x = (f[L_1I_1J_K0] + f[L_1I_J0_1K] + f[L_1I_J0_K0] +
                             f[L_1I_J0_K1] + f[L_1I_J1_K0])
            double den_x1 = (f[L_I1_1J_K0] + f[L_I1_J0_1K] + f[L_I1_J0_K0] +
                             f[L_I1_J0_K1] + f[L_I1_J1_K0])

            double den_x0_1y = f[L_I0_1J_1K] + f[L_I0_1J_K0] + f[L_I0_1J_K1]
            double den_x0_y1 = f[L_I0_J1_1K] + f[L_I0_J1_K0] + f[L_I0_J1_K1]

            double den = (den_1x + den_x1 + den_x0_1y + den_x0_y1 +
                          f[L_I0_J0_1K] + f[L_I0_J0_K0] + f[L_I0_J0_K1])
        
        mom[X] = den_x1 - den_1x
          
        mom[Y] = (f[L_1I_J1_K0] + den_x0_y1 + f[L_I1_J1_K0] -
                  f[L_1I_1J_K0] - den_x0_1y - f[L_I1_1J_K0])

        mom[Z] = (f[L_1I_J0_K1] + f[L_I0_1J_K1] + f[L_I0_J0_K1] +
                  f[L_I0_J1_K1] + f[L_I1_J0_K1] - f[L_1I_J0_1K] -
                  f[L_I0_1J_1K] - f[L_I0_J0_1K] - f[L_I0_J1_1K] -
                  f[L_I1_J0_1K])

        return den    

    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cdef inline double f_to_den_vel(self, double *f, double *vel) nogil:
        """Compute (mass) density and velocity from the distribution values.
        
        Parameters
        ----------
        f : double* (array, element count >= 19, read)
            distribution values
        vel : double* (array, element count >= 3, write)
              velocity vector

        Returns
        -------
        density : double
        """
        cdef:
            double den_1x = (f[L_1I_1J_K0] + f[L_1I_J0_1K] + f[L_1I_J0_K0] +
                             f[L_1I_J0_K1] + f[L_1I_J1_K0])
            double den_x1 = (f[L_I1_1J_K0] + f[L_I1_J0_1K] + f[L_I1_J0_K0] +
                             f[L_I1_J0_K1] + f[L_I1_J1_K0])

            double den_x0_1y = f[L_I0_1J_1K] + f[L_I0_1J_K0] + f[L_I0_1J_K1]
            double den_x0_y1 = f[L_I0_J1_1K] + f[L_I0_J1_K0] + f[L_I0_J1_K1]

            double den = (den_1x + den_x1 + den_x0_1y + den_x0_y1 +
                          f[L_I0_J0_1K] + f[L_I0_J0_K0] + f[L_I0_J0_K1])

            double jx = den_x1 - den_1x

            double jy = (f[L_1I_J1_K0] + den_x0_y1 + f[L_I1_J1_K0] -
                         f[L_1I_1J_K0] - den_x0_1y - f[L_I1_1J_K0])

            double jz = (f[L_1I_J0_K1] + f[L_I0_1J_K1] + f[L_I0_J0_K1] +
                         f[L_I0_J1_K1] + f[L_I1_J0_K1] - f[L_1I_J0_1K] -
                         f[L_I0_1J_1K] - f[L_I0_J0_1K] - f[L_I0_J1_1K] -
                         f[L_I1_J0_1K])
                            
            double inv_den = 1.0/den
        
        vel[X] = inv_den*jx
        vel[Y] = inv_den*jy
        vel[Z] = inv_den*jz

        return den    

    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cdef inline void f_to_h(self, double *f, double *h) nogil:
        """Compute Hermite polynomial moments from the distributions.
        
        Parameters
        ----------
        f : double* (array, element count >= 19, read)
            distribution values
        h : double* (array, element count >= 10, write)
            Hermite polynomial moments
        """
        cdef:
            int l, m
            double den_1x = (f[L_1I_1J_K0] + f[L_1I_J0_1K] + f[L_1I_J0_K0] +
                             f[L_1I_J0_K1] + f[L_1I_J1_K0])
            double den_x1 = (f[L_I1_1J_K0] + f[L_I1_J0_1K] + f[L_I1_J0_K0] +
                             f[L_I1_J0_K1] + f[L_I1_J1_K0])

            double den_x0_1y = f[L_I0_1J_1K] + f[L_I0_1J_K0] + f[L_I0_1J_K1]
            double den_x0_y1 = f[L_I0_J1_1K] + f[L_I0_J1_K0] + f[L_I0_J1_K1]

            double den_1y = f[L_1I_1J_K0] + den_x0_1y + f[L_I1_1J_K0]
            double den_y1 = f[L_1I_J1_K0] + den_x0_y1 + f[L_I1_J1_K0]

            double den_1z = (f[L_1I_J0_1K] + f[L_I0_1J_1K] + f[L_I0_J0_1K] +
                             f[L_I0_J1_1K] + f[L_I1_J0_1K])
            double den_z1 = (f[L_1I_J0_K1] + f[L_I0_1J_K1] + f[L_I0_J0_K1] +
                             f[L_I0_J1_K1] + f[L_I1_J0_K1])

            double den = (den_1x + den_x1 + den_x0_1y + den_x0_y1 +
                          f[L_I0_J0_1K] + f[L_I0_J0_K0] + f[L_I0_J0_K1])

            double jx = den_x1 - den_1x
            double jy = den_y1 - den_1y 
            double jz = den_z1 - den_1z

            double m2_xx = den_1x + den_x1 
            double m2_yy = den_1y + den_y1
            double m2_zz = den_1z + den_z1

            double m2_xy = (f[L_1I_1J_K0] + f[L_I1_J1_K0] - f[L_1I_J1_K0] -
                            f[L_I1_1J_K0])
                               
            double m2_xz = (f[L_1I_J0_1K] + f[L_I1_J0_K1] - f[L_1I_J0_K1] -
                            f[L_I1_J0_1K])

            double m2_yz = (f[L_I0_1J_1K] + f[L_I0_J1_K1] - f[L_I0_1J_K1] -
                            f[L_I0_J1_1K])
                
        h[H0] = den
        h[H1_X] = jx
        h[H1_Y] = jy
        h[H1_Z] = jz
        h[H2_XX] = m2_xx - self.INV_AS2*den
        h[H2_XY] = m2_xy
        h[H2_XZ] = m2_xz
        h[H2_YY] = m2_yy - self.INV_AS2*den
        h[H2_YZ] = m2_yz
        h[H2_ZZ] = m2_zz - self.INV_AS2*den
        
    # -----------------------------------------------------------------------
    # Functions for projecting moments into the kinetic space
    # (i.e. transformation of moments into distributions)
    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cdef inline void h_to_f(self, double *h, double *f) nogil:
        """Compute distributions from the Hermite polynomial moments
        (kinetic projection).
        
        Parameters
        ----------
        h : double* (array, element count >= 10, read)
            Hermite polynomial moments
        f : double* (array, element count >= 19, write)
            distribution values
        """
        cdef:
            double e_xy = 2*h[H0] + 6*h[H2_XX] + 6*h[H2_YY] - 3*h[H2_ZZ]

            double e_1x_1y = e_xy + 18*h[H2_XY]
            double o_1x_1y = -6*h[H1_X] - 6*h[H1_Y]
            
            double e_1x_y1 = e_xy - 18*h[H2_XY]
            double o_1x_y1 = -6*h[H1_X] + 6*h[H1_Y]

            double e_xz = 2*h[H0] + 6*h[H2_XX] - 3*h[H2_YY] + 6*h[H2_ZZ]
               
            double e_1x_1z = e_xz + 18*h[H2_XZ]
            double o_1x_1z = -6*h[H1_X] - 6*h[H1_Z]

            double e_1x_z1 = e_xz - 18*h[H2_XZ]
            double o_1x_z1 = -6*h[H1_X] + 6*h[H1_Z]

            double e_yz = 2*h[H0] - 3*h[H2_XX] + 6*h[H2_YY] + 6*h[H2_ZZ]
                
            double e_1y_1z = e_yz + 18*h[H2_YZ]
            double o_1y_1z = -6*h[H1_Y] - 6*h[H1_Z]

            double e_1y_z1 = e_yz - 18*h[H2_YZ]
            double o_1y_z1 = -6*h[H1_Y] + 6*h[H1_Z]

            double e_1x = 4*h[H0] + 12*h[H2_XX] - 6*h[H2_YY] - 6*h[H2_ZZ]
            double e_1y = 4*h[H0] - 6*h[H2_XX] + 12*h[H2_YY] - 6*h[H2_ZZ]
            double e_1z = 4*h[H0] - 6*h[H2_XX] - 6*h[H2_YY] + 12*h[H2_ZZ]

        f[L_1I_1J_K0] = (1.0/72.0)*(e_1x_1y + o_1x_1y)
        f[L_I1_J1_K0] = (1.0/72.0)*(e_1x_1y - o_1x_1y)

        f[L_1I_J0_1K] = (1.0/72.0)*(e_1x_1z + o_1x_1z)
        f[L_I1_J0_K1] = (1.0/72.0)*(e_1x_1z - o_1x_1z)

        f[L_1I_J0_K0] = (1.0/72.0)*(e_1x - 12*h[H1_X])
        f[L_I1_J0_K0] = (1.0/72.0)*(e_1x + 12*h[H1_X])

        f[L_1I_J0_K1] = (1.0/72.0)*(e_1x_z1 + o_1x_z1)
        f[L_I1_J0_1K] = (1.0/72.0)*(e_1x_z1 - o_1x_z1)

        f[L_1I_J1_K0] = (1.0/72.0)*(e_1x_y1 + o_1x_y1)
        f[L_I1_1J_K0] = (1.0/72.0)*(e_1x_y1 - o_1x_y1)

        f[L_I0_1J_1K] = (1.0/72.0)*(e_1y_1z + o_1y_1z)
        f[L_I0_J1_K1] = (1.0/72.0)*(e_1y_1z - o_1y_1z)

        f[L_I0_1J_K0] = (1.0/72.0)*(e_1y - 12*h[H1_Y])
        f[L_I0_J1_K0] = (1.0/72.0)*(e_1y + 12*h[H1_Y])

        f[L_I0_1J_K1] = (1.0/72.0)*(e_1y_z1 + o_1y_z1)
        f[L_I0_J1_1K] = (1.0/72.0)*(e_1y_z1 - o_1y_z1)

        f[L_I0_J0_1K] = (1.0/72.0)*(e_1z - 12*h[H1_Z])
        f[L_I0_J0_K1] = (1.0/72.0)*(e_1z + 12*h[H1_Z])

        f[L_I0_J0_K0] = (1.0/3.0)*h[H0] - 0.5*(h[H2_XX]+h[H2_YY]+h[H2_ZZ])

    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cdef inline void add_h_to_f(self, double *h, double *f) nogil:
        """Compute distributions from the Hermite polynomial moments
        (kinetic projection); add the results to the given distributions.
        
        Parameters
        ----------
        h : double* (array, element count >= 10, read)
            Hermite polynomial moments
        f : double* (array, element count >= 19, write/add)
            distribution values
        """
        cdef:
            double e_xy = 2*h[H0] + 6*h[H2_XX] + 6*h[H2_YY] - 3*h[H2_ZZ]

            double e_1x_1y = e_xy + 18*h[H2_XY]
            double o_1x_1y = -6*h[H1_X] - 6*h[H1_Y]
            
            double e_1x_y1 = e_xy - 18*h[H2_XY]
            double o_1x_y1 = -6*h[H1_X] + 6*h[H1_Y]

            double e_xz = 2*h[H0] + 6*h[H2_XX] - 3*h[H2_YY] + 6*h[H2_ZZ]
                
            double e_1x_1z = e_xz + 18*h[H2_XZ]
            double o_1x_1z = -6*h[H1_X] - 6*h[H1_Z]

            double e_1x_z1 = e_xz - 18*h[H2_XZ]
            double o_1x_z1 = -6*h[H1_X] + 6*h[H1_Z]

            double e_yz = 2*h[H0] - 3*h[H2_XX] + 6*h[H2_YY] + 6*h[H2_ZZ]
                
            double e_1y_1z = e_yz + 18*h[H2_YZ]
            double o_1y_1z = -6*h[H1_Y] - 6*h[H1_Z]

            double e_1y_z1 = e_yz - 18*h[H2_YZ]
            double o_1y_z1 = -6*h[H1_Y] + 6*h[H1_Z]

            double e_1x = 4*h[H0] + 12*h[H2_XX] - 6*h[H2_YY] - 6*h[H2_ZZ]
            double e_1y = 4*h[H0] - 6*h[H2_XX] + 12*h[H2_YY] - 6*h[H2_ZZ]
            double e_1z = 4*h[H0] - 6*h[H2_XX] - 6*h[H2_YY] + 12*h[H2_ZZ]

        f[L_1I_1J_K0] += (1.0/72.0)*(e_1x_1y + o_1x_1y)
        f[L_I1_J1_K0] += (1.0/72.0)*(e_1x_1y - o_1x_1y)

        f[L_1I_J0_1K] += (1.0/72.0)*(e_1x_1z + o_1x_1z)
        f[L_I1_J0_K1] += (1.0/72.0)*(e_1x_1z - o_1x_1z)

        f[L_1I_J0_K0] += (1.0/72.0)*(e_1x - 12*h[H1_X])
        f[L_I1_J0_K0] += (1.0/72.0)*(e_1x + 12*h[H1_X])

        f[L_1I_J0_K1] += (1.0/72.0)*(e_1x_z1 + o_1x_z1)
        f[L_I1_J0_1K] += (1.0/72.0)*(e_1x_z1 - o_1x_z1)

        f[L_1I_J1_K0] += (1.0/72.0)*(e_1x_y1 + o_1x_y1)
        f[L_I1_1J_K0] += (1.0/72.0)*(e_1x_y1 - o_1x_y1)

        f[L_I0_1J_1K] += (1.0/72.0)*(e_1y_1z + o_1y_1z)
        f[L_I0_J1_K1] += (1.0/72.0)*(e_1y_1z - o_1y_1z)

        f[L_I0_1J_K0] += (1.0/72.0)*(e_1y - 12*h[H1_Y])
        f[L_I0_J1_K0] += (1.0/72.0)*(e_1y + 12*h[H1_Y])

        f[L_I0_1J_K1] += (1.0/72.0)*(e_1y_z1 + o_1y_z1)
        f[L_I0_J1_1K] += (1.0/72.0)*(e_1y_z1 - o_1y_z1)

        f[L_I0_J0_1K] += (1.0/72.0)*(e_1z - 12*h[H1_Z])
        f[L_I0_J0_K1] += (1.0/72.0)*(e_1z + 12*h[H1_Z])

        f[L_I0_J0_K0] += (1.0/3.0)*h[H0] - 0.5*(h[H2_XX]+h[H2_YY]+h[H2_ZZ])

    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cdef inline void den_vel_to_heq(self, double den, double *vel,
                                    double *heq) nogil:
        """Compute Hermite polynomial equilibrium moments
        from the density and velocity.
       
        Parameters
        ----------
        den : double
              density
        vel : double* (array, element count >= 3, read)
              velocity vector
        heq : double* (array, element count >= 10, write)
              Hermite polynomial equilibrium moments
        """
        cdef:
            double jx = den*vel[X], jy = den*vel[Y], jz = den*vel[Z]
                
        heq[H0] = den
        heq[H1_X] = jx
        heq[H1_Y] = jy
        heq[H1_Z] = jz
        heq[H2_XX] = jx*vel[X]
        heq[H2_XY] = jx*vel[Y]
        heq[H2_XZ] = jx*vel[Z]
        heq[H2_YY] = jy*vel[Y]
        heq[H2_YZ] = jy*vel[Z]
        heq[H2_ZZ] = jz*vel[Z]

    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cdef void eq(self, double den, double *vel, double *feq) nogil:
        """Compute equilibrium distributions from the density and velocity.
       
        Parameters
        ----------
        den : double
              density
        vel : double* (array, element count >= 3, read)
              velocity vector
        feq : double* (array, element count >= 19, write)
              equilibrium distributions
        """
        cdef double heq[HMOMS]
        
        self.den_vel_to_heq(den, vel, heq)
        self.h_to_f(heq, feq)

    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cdef double forw_euler_BGK(self, double *f, double *acc, double *prm,
                               double *vel) nogil:
        """Forward-Euler update procedure with the BGK relaxation scheme.
        
        Parameters
        ----------
        f : double* (array, element count >= 19, read and write)
            distribution values
        acc : double* (array, element count >= 3, read)
              acceleration vector
        prm : double* (array, element count >= 1, read)
              relaxation parameters (inverse of relaxation times)
              prm[0] = a common relaxation parameter for all moments
        vel : double* (array, element count >= 3, write)
              velocity vector

        Returns
        -------
        density : double
        """
        cdef:
            double den, inv_tau = prm[0], rlx_cff1 = 1.0 - inv_tau
            double rlx_cff2 = (1.0 - 0.5*inv_tau)
            double jx, jy, jz
            double h[HMOMS]

        # Compute equilibrium moments
        den = self.f_to_den_vel(f,vel)
        vel[X] += 0.5*acc[X]
        vel[Y] += 0.5*acc[Y]
        vel[Z] += 0.5*acc[Z]
        self.den_vel_to_heq(den, vel, h)
                            
        h[H0] *= inv_tau
        h[H1_X] *= inv_tau
        h[H1_Y] *= inv_tau
        h[H1_Z] *= inv_tau
        h[H2_XX] *= inv_tau
        h[H2_XY] *= inv_tau
        h[H2_XZ] *= inv_tau
        h[H2_YY] *= inv_tau
        h[H2_YZ] *= inv_tau
        h[H2_ZZ] *= inv_tau

        # Compute acceleration moments
        jx = den*vel[X]
        jy = den*vel[Y]
        jz = den*vel[Z]
            
        h[H1_X] += rlx_cff2*acc[X]*den
        h[H1_Y] += rlx_cff2*acc[Y]*den
        h[H1_Z] += rlx_cff2*acc[Z]*den
            
        h[H2_XX] += 2.0*rlx_cff2*acc[X]*jx
        h[H2_XY] += rlx_cff2*(acc[X]*jy + acc[Y]*jx)
        h[H2_XZ] += rlx_cff2*(acc[X]*jz + acc[Z]*jx)
        h[H2_YY] += 2.0*rlx_cff2*acc[Y]*jy
        h[H2_YZ] += rlx_cff2*(acc[Y]*jz + acc[Z]*jy)
        h[H2_ZZ] += 2.0*rlx_cff2*acc[Z]*jz

        # Relaxation of the incoming distributions
        f[L_1I_1J_K0] *= rlx_cff1
        f[L_1I_J0_1K] *= rlx_cff1
        f[L_1I_J0_K0] *= rlx_cff1
        f[L_1I_J0_K1] *= rlx_cff1
        f[L_1I_J1_K0] *= rlx_cff1
        f[L_I0_1J_1K] *= rlx_cff1
        f[L_I0_1J_K0] *= rlx_cff1
        f[L_I0_1J_K1] *= rlx_cff1
        f[L_I0_J0_1K] *= rlx_cff1
        f[L_I0_J0_K0] *= rlx_cff1
        f[L_I0_J0_K1] *= rlx_cff1
        f[L_I0_J1_1K] *= rlx_cff1
        f[L_I0_J1_K0] *= rlx_cff1
        f[L_I0_J1_K1] *= rlx_cff1
        f[L_I1_1J_K0] *= rlx_cff1
        f[L_I1_J0_1K] *= rlx_cff1
        f[L_I1_J0_K0] *= rlx_cff1
        f[L_I1_J0_K1] *= rlx_cff1
        f[L_I1_J1_K0] *= rlx_cff1

        # Add the equilibrium and acceleration moments
        # to the relaxed incoming distributions (kinetic projection)
        self.add_h_to_f(h, f)
        
        return den
        
    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    cdef double forw_euler_TRT(self, double *f, double *acc, double *prm,
                               double *vel) nogil:
        """Forward-Euler update procedure with the TRT relaxation scheme.
        
        Parameters
        ----------
        f : double* (array, element count >= 19, read and write)
            distribution values
        acc : double* (array, element count >= 3, read)
              acceleration vector
        prm : double* (array, element count >= 2, read)
              relaxation parameters (inverse of relaxation times)
              prm[0] = relaxation parameter for the even moments
              prm[1] = relaxation parameter for the odd moments
        vel : double* (array, element count >= 3, write)
              velocity vector

        Returns
        -------
        density : double
        """
        cdef:
            double faux1,faux2,faux3,faux4,faux5,faux6,faux7,faux8,faux9
            double inv_tau_e = prm[0], inv_tau_o = prm[1]
            double rlx_cff1 = 1.0 - 0.5*inv_tau_o
            double rlx_cff2 = 1.0 - 0.5*inv_tau_e
            double rlx_cff3 = 1.0 - 0.5*(inv_tau_e + inv_tau_o)
            double rlx_cff4 = 0.5*(inv_tau_e - inv_tau_o)
            double den, jx, jy, jz
            double h[HMOMS]
                
        # Compute equilibrium moments
        den = self.f_to_den_vel(f,vel)
        vel[X] += 0.5*acc[X]
        vel[Y] += 0.5*acc[Y]
        vel[Z] += 0.5*acc[Z]
        self.den_vel_to_heq(den, vel, h)
                            
        h[H0] *= inv_tau_e
        h[H1_X] *= inv_tau_o
        h[H1_Y] *= inv_tau_o
        h[H1_Z] *= inv_tau_o
        h[H2_XX] *= inv_tau_e
        h[H2_XY] *= inv_tau_e
        h[H2_XZ] *= inv_tau_e
        h[H2_YY] *= inv_tau_e
        h[H2_YZ] *= inv_tau_e
        h[H2_ZZ] *= inv_tau_e

        # Compute acceleration moments
        jx = den*vel[X]
        jy = den*vel[Y]
        jz = den*vel[Z]
            
        h[H1_X] += rlx_cff1*acc[X]*den
        h[H1_Y] += rlx_cff1*acc[Y]*den
        h[H1_Z] += rlx_cff1*acc[Z]*den
            
        h[H2_XX] += 2.0*rlx_cff2*acc[X]*jx
        h[H2_XY] += rlx_cff2*(acc[X]*jy + acc[Y]*jx)
        h[H2_XZ] += rlx_cff2*(acc[X]*jz + acc[Z]*jx)
        h[H2_YY] += 2.0*rlx_cff2*acc[Y]*jy
        h[H2_YZ] += rlx_cff2*(acc[Y]*jz + acc[Z]*jy)
        h[H2_ZZ] += 2.0*rlx_cff2*acc[Z]*jz
            
        # Relaxation of the incoming distributions
        faux1 = f[L_1I_1J_K0]
        f[L_1I_1J_K0] = rlx_cff3*faux1 - rlx_cff4*f[L_I1_J1_K0]
        f[L_I1_J1_K0] = rlx_cff3*f[L_I1_J1_K0] - rlx_cff4*faux1

        faux2 = f[L_1I_J0_1K]
        f[L_1I_J0_1K] = rlx_cff3*faux2 - rlx_cff4*f[L_I1_J0_K1]
        f[L_I1_J0_K1] = rlx_cff3*f[L_I1_J0_K1] - rlx_cff4*faux2

        faux3 = f[L_1I_J0_K0]
        f[L_1I_J0_K0] = rlx_cff3*faux3 - rlx_cff4*f[L_I1_J0_K0]
        f[L_I1_J0_K0] = rlx_cff3*f[L_I1_J0_K0] - rlx_cff4*faux3

        faux4 = f[L_1I_J0_K1]
        f[L_1I_J0_K1] = rlx_cff3*faux4 - rlx_cff4*f[L_I1_J0_1K]
        f[L_I1_J0_1K] = rlx_cff3*f[L_I1_J0_1K] - rlx_cff4*faux4

        faux5 = f[L_1I_J1_K0]
        f[L_1I_J1_K0] = rlx_cff3*faux5 - rlx_cff4*f[L_I1_1J_K0]
        f[L_I1_1J_K0] = rlx_cff3*f[L_I1_1J_K0] - rlx_cff4*faux5

        faux6 = f[L_I0_1J_1K]
        f[L_I0_1J_1K] = rlx_cff3*faux6 - rlx_cff4*f[L_I0_J1_K1]
        f[L_I0_J1_K1] = rlx_cff3*f[L_I0_J1_K1] - rlx_cff4*faux6

        faux7 = f[L_I0_1J_K0]
        f[L_I0_1J_K0] = rlx_cff3*faux7 - rlx_cff4*f[L_I0_J1_K0]
        f[L_I0_J1_K0] = rlx_cff3*f[L_I0_J1_K0] - rlx_cff4*faux7

        faux8 = f[L_I0_1J_K1]
        f[L_I0_1J_K1] = rlx_cff3*faux8 - rlx_cff4*f[L_I0_J1_1K]
        f[L_I0_J1_1K] = rlx_cff3*f[L_I0_J1_1K] - rlx_cff4*faux8

        faux9 = f[L_I0_J0_1K]
        f[L_I0_J0_1K] = rlx_cff3*faux9 - rlx_cff4*f[L_I0_J0_K1]
        f[L_I0_J0_K1] = rlx_cff3*f[L_I0_J0_K1] - rlx_cff4*faux9

        f[L_I0_J0_K0] *= (1.0 - inv_tau_e)

        # Add the equilibrium and acceleration moments
        # to the relaxed incoming distributions (kinetic projection)
        self.add_h_to_f(h, f)
        
        return den

    # -----------------------------------------------------------------------
    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cdef double forw_euler_regul(self, double *f, double *acc, double *prm,
                                 double *vel) nogil:
        """Forward-Euler update procedure with the regularization
        (and the single-relaxation time scheme).
        
        Parameters
        ----------
        f : double* (array, element count >= 19, read and write)
            distribution values
        acc : double* (array, element count >= 3, read)
              acceleration vector
        prm : double* (array, element count >= 1, read)
              relaxation parameters (inverse of relaxation times)
              prm[0] = a common relaxation parameter for all moments
        vel : double* (array, element count >= 3, write)
              velocity vector

        Returns
        -------
        density : double
        """
        cdef:
            double den, inv_den, inv_tau = prm[0], rlx_cff1 = 1.0 - inv_tau
            double rlx_cff2 = (1.0 - 0.5*inv_tau)
            double jx, jy, jz
            double heq[HMOMS]
            double h[HMOMS]

        # Compute all moments
        self.f_to_h(f, h)
     
        # Compute equilibrium moments
        den = h[H0]
        inv_den = 1.0/den
        jx = h[H1_X] + 0.5*acc[X]*den
        jy = h[H1_Y] + 0.5*acc[Y]*den
        jz = h[H1_Z] + 0.5*acc[Z]*den
        vel[X] = inv_den*jx
        vel[Y] = inv_den*jy
        vel[Z] = inv_den*jz

        self.den_vel_to_heq(den, vel, heq)
       
        # Relaxation of the full moments
        h[H0]    *= rlx_cff1
        h[H1_X]  *= rlx_cff1
        h[H1_Y]  *= rlx_cff1
        h[H1_Z]  *= rlx_cff1
        h[H2_XX] *= rlx_cff1
        h[H2_XY] *= rlx_cff1
        h[H2_XZ] *= rlx_cff1
        h[H2_YY] *= rlx_cff1
        h[H2_YZ] *= rlx_cff1
        h[H2_ZZ] *= rlx_cff1
            
        # Relaxation of the equilibrium moments
        h[H0]    += inv_tau*heq[H0]
        h[H1_X]  += inv_tau*heq[H1_X]
        h[H1_Y]  += inv_tau*heq[H1_Y]
        h[H1_Z]  += inv_tau*heq[H1_Z]
        h[H2_XX] += inv_tau*heq[H2_XX]
        h[H2_XY] += inv_tau*heq[H2_XY]
        h[H2_XZ] += inv_tau*heq[H2_XZ]
        h[H2_YY] += inv_tau*heq[H2_YY]
        h[H2_YZ] += inv_tau*heq[H2_YZ]
        h[H2_ZZ] += inv_tau*heq[H2_ZZ]

        # Compute acceleration moments
        h[H1_X]  += rlx_cff2*acc[X]*den
        h[H1_Y]  += rlx_cff2*acc[Y]*den
        h[H1_Z]  += rlx_cff2*acc[Z]*den
        h[H2_XX] += 2.0*rlx_cff2*acc[X]*jx
        h[H2_XY] += rlx_cff2*(acc[X]*jy + acc[Y]*jx)
        h[H2_XZ] += rlx_cff2*(acc[X]*jz + acc[Z]*jx)
        h[H2_YY] += 2.0*rlx_cff2*acc[Y]*jy
        h[H2_YZ] += rlx_cff2*(acc[Y]*jz + acc[Z]*jy)
        h[H2_ZZ] += 2.0*rlx_cff2*acc[Z]*jz

        # Post-collisional distributions (kinetic projection of the moments)
        self.h_to_f(h, f)
        
        return den
            
    # -----------------------------------------------------------------------
