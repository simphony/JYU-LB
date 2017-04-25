"""Lattice-Boltzmann boundary conditions for the D3Q19 discrete velocity set.

Details
-------
Fixed density and velocity bcs enforced using the Zou & He scheme.

Author
------
Keijo Mattila, JYU, April 2017.
"""
# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
import cython
cimport cython
from cython cimport boundscheck, wraparound

from D3Q19 cimport L_1I_1J_K0, L_1I_J0_1K, L_1I_J0_K0, L_1I_J0_K1, L_1I_J1_K0
from D3Q19 cimport L_I0_1J_1K, L_I0_1J_K0, L_I0_1J_K1
from D3Q19 cimport L_I0_J0_1K, L_I0_J0_K0, L_I0_J0_K1
from D3Q19 cimport L_I0_J1_1K, L_I0_J1_K0, L_I0_J1_K1
from D3Q19 cimport L_I1_1J_K0, L_I1_J0_1K, L_I1_J0_K0, L_I1_J0_K1, L_I1_J1_K0

from defs cimport X,Y,Z

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
cdef void fix_den_face_1x_D3Q19(double den, double *vel, double *f) nogil:
    """Enforce density at the boundary with outward unit normal = (-1,0,0).
    Also the velocity components parallel to the boundary are enforced.
    The boundary condition scheme by Zou & He is applied.
    
    Parameters
    ----------
    den : double
          density, enforced
    vel : double* (array, element count >= 3, read)
          vel[X] = velocity component perpendicular to the boundary, not used
          vel[Y] = velocity component parallel to the boundary, enforced
          vel[Z] = velocity component parallel to the boundary, enforced
    f : double* (array, element count >= 19, read and write)
        distribution values associated with the lattice vectors
        L_I1_1J_K0, L_I1_J0_1K, L_I1_J0_K0, L_I1_J0_K1, L_I1_J1_K0
        are assigned
    """
    cdef:
        double cr = 1.0, inv_cr = 1.0

        double den_kn = (f[L_I0_J0_K0] + f[L_1I_J0_K0] + f[L_I0_1J_K0] +
                         f[L_I0_J1_K0] + f[L_I0_J0_1K] + f[L_I0_1J_1K] +
                         f[L_I0_J1_1K] + f[L_1I_J0_1K] + f[L_1I_1J_K0] +
                         f[L_1I_J1_K0] + f[L_I0_J0_K1] + f[L_I0_1J_K1] +
                         f[L_I0_J1_K1] + f[L_1I_J0_K1])

        double jx_pos = cr*(den - den_kn)
        double jx_neg = cr*(f[L_1I_J0_K0] + f[L_1I_J0_1K] + f[L_1I_1J_K0] +
                            f[L_1I_J1_K0] + f[L_1I_J0_K1])

        double jx = inv_cr*(jx_pos - jx_neg)
        double jy = inv_cr*den*vel[Y], jz = inv_cr*den*vel[Z]

    _assign_f_1x(jx, jy, jz, f)

# ---------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
cdef void fix_den_face_x1_D3Q19(double den, double *vel, double *f) nogil:
    """Enforce density at the boundary with outward unit normal = (1,0,0).
    Also the velocity components parallel to the boundary are enforced.
    The boundary condition scheme by Zou & He is applied.
    
    Parameters
    ----------
    den : double
          density, enforced
    vel : double* (array, element count >= 3, read)
          vel[X] = velocity component perpendicular to the boundary, not used
          vel[Y] = velocity component parallel to the boundary, enforced
          vel[Z] = velocity component parallel to the boundary, enforced
    f : double* (array, element count >= 19, read and write)
        distribution values associated with the lattice vectors
        L_1I_1J_K0, L_1I_J0_1K, L_1I_J0_K0, L_1I_J0_K1, L_1I_J1_K0
        are assigned
    """
    cdef:
        double cr = 1.0, inv_cr = 1.0

        double den_kn = (f[L_I0_J0_K0] + f[L_I1_J0_K0] + f[L_I0_1J_K0] +
                         f[L_I0_J1_K0] + f[L_I0_J0_1K] + f[L_I0_1J_1K] +
                         f[L_I0_J1_1K] + f[L_I1_J0_1K] + f[L_I1_1J_K0] +
                         f[L_I1_J1_K0] + f[L_I0_J0_K1] + f[L_I0_1J_K1] +
                         f[L_I0_J1_K1] + f[L_I1_J0_K1])

        double jx_neg = cr*(den - den_kn)
        double jx_pos = cr*(f[L_I1_J0_K0] + f[L_I1_J0_1K] + f[L_I1_1J_K0] +
                            f[L_I1_J1_K0] + f[L_I1_J0_K1]),
        
        double jx = inv_cr*(jx_pos - jx_neg)
        double jy = inv_cr*den*vel[Y], jz = inv_cr*den*vel[Z]
                 
    _assign_f_x1(jx, jy, jz, f)

# ---------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
cdef void fix_den_face_1y_D3Q19(double den, double *vel, double *f) nogil:
    """Enforce density at the boundary with outward unit normal = (0,-1,0).
    Also the velocity components parallel to the boundary are enforced.
    The boundary condition scheme by Zou & He is applied.
    
    Parameters
    ----------
    den : double
          density, enforced
    vel : double* (array, element count >= 3, read)
          vel[X] = velocity component parallel to the boundary, enforced
          vel[Y] = velocity component perpendicular to the boundary, not used
          vel[Z] = velocity component parallel to the boundary, enforced
    f : double* (array, element count >= 19, read and write)
        distribution values associated with the lattice vectors
        L_1I_J1_K0, L_I0_J1_1K, L_I0_J1_K0, L_I0_J1_K1, L_I1_J1_K0
        are assigned
    """
    cdef:
        double cr = 1.0, inv_cr = 1.0

        double den_kn = (f[L_I0_J0_K0] + f[L_1I_J0_K0] + f[L_I1_J0_K0] +
                         f[L_I0_1J_K0] + f[L_I0_J0_1K] + f[L_I0_1J_1K] +
                         f[L_1I_J0_1K] + f[L_I1_J0_1K] + f[L_1I_1J_K0] +
                         f[L_I1_1J_K0] + f[L_I0_J0_K1] + f[L_I0_1J_K1] +
                         f[L_1I_J0_K1] + f[L_I1_J0_K1])

        double jy_pos = cr*(den - den_kn)
        double jy_neg = cr*(f[L_I0_1J_K0] + f[L_I0_1J_1K] + f[L_1I_1J_K0] +
                            f[L_I1_1J_K0] + f[L_I0_1J_K1])
                
        double jy = inv_cr*(jy_pos - jy_neg)
        double jx = inv_cr*den*vel[X], jz = inv_cr*den*vel[Z]
                 
    _assign_f_1y(jx, jy, jz, f)
    
# ---------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
cdef void fix_den_face_y1_D3Q19(double den, double *vel, double *f) nogil:
    """Enforce density at the boundary with outward unit normal = (0,1,0).
    Also the velocity components parallel to the boundary are enforced.
    The boundary condition scheme by Zou & He is applied.
    
    Parameters
    ----------
    den : double
          density, enforced
    vel : double* (array, element count >= 3, read)
          vel[X] = velocity component parallel to the boundary, enforced
          vel[Y] = velocity component perpendicular to the boundary, not used
          vel[Z] = velocity component parallel to the boundary, enforced
    f : double* (array, element count >= 19, read and write)
        distribution values associated with the lattice vectors
        L_1I_1J_K0, L_I0_1J_1K, L_I0_1J_K0, L_I0_1J_K1, L_I1_1J_K0
        are assigned
    """
    cdef:
        double cr = 1.0, inv_cr = 1.0

        double den_kn = (f[L_I0_J0_K0] + f[L_1I_J0_K0] + f[L_I1_J0_K0] +
                         f[L_I0_J1_K0] + f[L_I0_J0_1K] + f[L_I0_J1_1K] +
                         f[L_1I_J0_1K] + f[L_I1_J0_1K] + f[L_1I_J1_K0] +
                         f[L_I1_J1_K0] + f[L_I0_J0_K1] + f[L_I0_J1_K1] +
                         f[L_1I_J0_K1] + f[L_I1_J0_K1])

        double jy_neg = cr*(den - den_kn)
        double jy_pos = cr*(f[L_I0_J1_K0] + f[L_I0_J1_1K] + f[L_1I_J1_K0] +
                            f[L_I1_J1_K0] + f[L_I0_J1_K1])
        
        double jy = inv_cr*(jy_pos - jy_neg)
        double jx = inv_cr*den*vel[X], jz = inv_cr*den*vel[Z]
                 
    _assign_f_y1(jx, jy, jz, f)

# ---------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
cdef void fix_den_face_1z_D3Q19(double den, double *vel, double *f) nogil:
    """Enforce density at the boundary with outward unit normal = (0,0,-1).
    Also the velocity components parallel to the boundary are enforced.
    The boundary condition scheme by Zou & He is applied.
    
    Parameters
    ----------
    den : double
          density, enforced
    vel : double* (array, element count >= 3, read)
          vel[X] = velocity component parallel to the boundary, enforced
          vel[Y] = velocity component parallel to the boundary, enforced
          vel[Z] = velocity component perpendicular to the boundary, not used
    f : double* (array, element count >= 19, read and write)
        distribution values associated with the lattice vectors
        L_1I_J0_K1, L_I0_1J_K1, L_I0_J0_K1, L_I0_J1_K1, L_I1_J0_K1
        are assigned
    """
    cdef:
        double cr = 1.0, inv_cr = 1.0

        double den_kn = (f[L_I0_J0_K0] + f[L_1I_J0_K0] + f[L_I1_J0_K0] +
                         f[L_I0_1J_K0] + f[L_I0_J1_K0] + f[L_I0_J0_1K] +
                         f[L_I0_1J_1K] + f[L_I0_J1_1K] + f[L_1I_J0_1K] +
                         f[L_I1_J0_1K] + f[L_1I_1J_K0] + f[L_I1_J1_K0] +
                         f[L_I1_1J_K0] + f[L_1I_J1_K0])

        double jz_pos = cr*(den - den_kn)
        double jz_neg = cr*(f[L_I0_J0_1K] + f[L_I0_1J_1K] + f[L_I0_J1_1K] +
                            f[L_1I_J0_1K] + f[L_I1_J0_1K])
        
        double jz = inv_cr*(jz_pos - jz_neg)
        double jx = inv_cr*den*vel[X], jy = inv_cr*den*vel[Y]
                 
    _assign_f_1z(jx, jy, jz, f)
    
# ---------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
cdef void fix_den_face_z1_D3Q19(double den, double *vel, double *f) nogil:
    """Enforce density at the boundary with outward unit normal = (0,0,1).
    Also the velocity components parallel to the boundary are enforced.
    The boundary condition scheme by Zou & He is applied.
    
    Parameters
    ----------
    den : double
          density, enforced
    vel : double* (array, element count >= 3, read)
          vel[X] = velocity component parallel to the boundary, enforced
          vel[Y] = velocity component parallel to the boundary, enforced
          vel[Z] = velocity component perpendicular to the boundary, not used
    f : double* (array, element count >= 19, read and write)
        distribution values associated with the lattice vectors
        L_1I_J0_1K, L_I0_1J_1K, L_I0_J0_1K, L_I0_J1_1K, L_I1_J0_1K
        are assigned
    """
    cdef:
        double cr = 1.0, inv_cr = 1.0

        double den_kn = (f[L_I0_J0_K0] + f[L_1I_J0_K0] + f[L_I1_J0_K0] +
                         f[L_I0_1J_K0] + f[L_I0_J1_K0] + f[L_I0_J0_K1] +
                         f[L_I0_1J_K1] + f[L_I0_J1_K1] + f[L_1I_J0_K1] +
                         f[L_I1_J0_K1] + f[L_1I_1J_K0] + f[L_I1_J1_K0] +
                         f[L_I1_1J_K0] + f[L_1I_J1_K0])
      
        double jz_neg = cr*(den - den_kn)
        double jz_pos = cr*(f[L_I0_J0_K1] + f[L_I0_1J_K1] + f[L_I0_J1_K1] +
                            f[L_1I_J0_K1] + f[L_I1_J0_K1])
        
        double jz = inv_cr*(jz_pos - jz_neg)
        double jx = inv_cr*den*vel[X], jy = inv_cr*den*vel[Y]
                 
    _assign_f_z1(jx, jy, jz, f)
    
# ---------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
cdef void fix_vel_face_1x_D3Q19(double den, double *vel, double *f) nogil:
    """Enforce velocity at the boundary with outward unit normal = (-1,0,0).
    The boundary condition scheme by Zou & He is applied.
    
    Parameters
    ----------
    den : double
          density, not used
    vel : double* (array, element count >= 3, read)
          velocity, enforced
    f : double* (array, element count >= 19, read and write)
        distribution values associated with the lattice vectors
        L_I1_1J_K0, L_I1_J0_1K, L_I1_J0_K0, L_I1_J0_K1, L_I1_J1_K0
        are assigned
    """
    cdef:
        double cr = 1.0, inv_cr = 1.0

        double den_kn = (f[L_I0_J0_K0] + f[L_1I_J0_K0] + f[L_I0_1J_K0] +
                         f[L_I0_J1_K0] + f[L_I0_J0_1K] + f[L_I0_1J_1K] +
                         f[L_I0_J1_1K] + f[L_1I_J0_1K] + f[L_1I_1J_K0] +
                         f[L_1I_J1_K0] + f[L_I0_J0_K1] + f[L_I0_1J_K1] +
                         f[L_I0_J1_K1] + f[L_1I_J0_K1])

        double jx_neg = cr*(f[L_1I_J0_K0] + f[L_1I_J0_1K] + f[L_1I_1J_K0] +
                            f[L_1I_J1_K0] + f[L_1I_J0_K1])
        double den_bc = (cr*den_kn + jx_neg)/(cr - vel[X])

        double jx = den_bc*inv_cr*vel[X]
        double jy = den_bc*inv_cr*vel[Y]
        double jz = den_bc*inv_cr*vel[Z]
                
    _assign_f_1x(jx, jy, jz, f)

# ---------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
cdef void fix_vel_face_x1_D3Q19(double den, double *vel, double *f) nogil:
    """Enforce velocity at the boundary with outward unit normal = (1,0,0).
    The boundary condition scheme by Zou & He is applied.
    
    Parameters
    ----------
    den : double
          density, not used
    vel : double* (array, element count >= 3, read)
          velocity, enforced
    f : double* (array, element count >= 19, read and write)
        distribution values associated with the lattice vectors
        L_1I_1J_K0, L_1I_J0_1K, L_1I_J0_K0, L_1I_J0_K1, L_1I_J1_K0
        are assigned
    """
    cdef:
        double cr = 1.0, inv_cr = 1.0

        double den_kn = (f[L_I0_J0_K0] + f[L_I1_J0_K0] + f[L_I0_1J_K0] +
                         f[L_I0_J1_K0] + f[L_I0_J0_1K] + f[L_I0_1J_1K] +
                         f[L_I0_J1_1K] + f[L_I1_J0_1K] + f[L_I1_1J_K0] +
                         f[L_I1_J1_K0] + f[L_I0_J0_K1] + f[L_I0_1J_K1] +
                         f[L_I0_J1_K1] + f[L_I1_J0_K1])

        double jx_pos = cr*(f[L_I1_J0_K0] + f[L_I1_J0_1K] + f[L_I1_1J_K0] +
                            f[L_I1_J1_K0] + f[L_I1_J0_K1])
        double den_bc = (cr*den_kn + jx_pos)/(cr + vel[X])

        double jx = den_bc*inv_cr*vel[X]
        double jy = den_bc*inv_cr*vel[Y]
        double jz = den_bc*inv_cr*vel[Z]
                 
    _assign_f_x1(jx, jy, jz, f)

# ---------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
cdef void fix_vel_face_1y_D3Q19(double den, double *vel, double *f) nogil:
    """Enforce velocity at the boundary with outward unit normal = (0,-1,0).
    The boundary condition scheme by Zou & He is applied.
    
    Parameters
    ----------
    den : double
          density, not used
    vel : double* (array, element count >= 3, read)
          velocity, enforced
    f : double* (array, element count >= 19, read and write)
        distribution values associated with the lattice vectors
        L_1I_J1_K0, L_I0_J1_1K, L_I0_J1_K0, L_I0_J1_K1, L_I1_J1_K0
        are assigned
    """
    cdef:
        double cr = 1.0, inv_cr = 1.0

        double den_kn = (f[L_I0_J0_K0] + f[L_1I_J0_K0] + f[L_I1_J0_K0] +
                         f[L_I0_1J_K0] + f[L_I0_J0_1K] + f[L_I0_1J_1K] +
                         f[L_1I_J0_1K] + f[L_I1_J0_1K] + f[L_1I_1J_K0] +
                         f[L_I1_1J_K0] + f[L_I0_J0_K1] + f[L_I0_1J_K1] +
                         f[L_1I_J0_K1] + f[L_I1_J0_K1])

        double jy_neg = cr*(f[L_I0_1J_K0] + f[L_I0_1J_1K] + f[L_1I_1J_K0] +
                            f[L_I1_1J_K0] + f[L_I0_1J_K1])
        double den_bc = (cr*den_kn + jy_neg)/(cr - vel[Y])
        
        double jx = den_bc*inv_cr*vel[X]
        double jy = den_bc*inv_cr*vel[Y]
        double jz = den_bc*inv_cr*vel[Z]
                 
    _assign_f_1y(jx, jy, jz, f)
    
# ---------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
cdef void fix_vel_face_y1_D3Q19(double den, double *vel, double *f) nogil:
    """Enforce velocity at the boundary with outward unit normal = (0,1,0).
    The boundary condition scheme by Zou & He is applied.
    
    Parameters
    ----------
    den : double
          density, not used
    vel : double* (array, element count >= 3, read)
          velocity, enforced
    f : double* (array, element count >= 19, read and write)
        distribution values associated with the lattice vectors
        L_1I_1J_K0, L_I0_1J_1K, L_I0_1J_K0, L_I0_1J_K1, L_I1_1J_K0
        are assigned
    """
    cdef:
        double cr = 1.0, inv_cr = 1.0

        double den_kn = (f[L_I0_J0_K0] + f[L_1I_J0_K0] + f[L_I1_J0_K0] +
                         f[L_I0_J1_K0] + f[L_I0_J0_1K] + f[L_I0_J1_1K] +
                         f[L_1I_J0_1K] + f[L_I1_J0_1K] + f[L_1I_J1_K0] +
                         f[L_I1_J1_K0] + f[L_I0_J0_K1] + f[L_I0_J1_K1] +
                         f[L_1I_J0_K1] + f[L_I1_J0_K1])

        double jy_pos = cr*(f[L_I0_J1_K0] + f[L_I0_J1_1K] + f[L_1I_J1_K0] +
                            f[L_I1_J1_K0] + f[L_I0_J1_K1])
        double den_bc = (cr*den_kn + jy_pos)/(cr + vel[Y])
        
        double jx = den_bc*inv_cr*vel[X]
        double jy = den_bc*inv_cr*vel[Y]
        double jz = den_bc*inv_cr*vel[Z]
                 
    _assign_f_y1(jx, jy, jz, f)

# ---------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
cdef void fix_vel_face_1z_D3Q19(double den, double *vel, double *f) nogil:
    """Enforce velocity at the boundary with outward unit normal = (0,0,-1).
    The boundary condition scheme by Zou & He is applied.
    
    Parameters
    ----------
    den : double
          density, not used
    vel : double* (array, element count >= 3, read)
          velocity, enforced
    f : double* (array, element count >= 19, read and write)
        distribution values associated with the lattice vectors
        L_1I_J0_K1, L_I0_1J_K1, L_I0_J0_K1, L_I0_J1_K1, L_I1_J0_K1
        are assigned
    """
    cdef:
        double cr = 1.0, inv_cr = 1.0

        double den_kn = (f[L_I0_J0_K0] + f[L_1I_J0_K0] + f[L_I1_J0_K0] +
                         f[L_I0_1J_K0] + f[L_I0_J1_K0] + f[L_I0_J0_1K] +
                         f[L_I0_1J_1K] + f[L_I0_J1_1K] + f[L_1I_J0_1K] +
                         f[L_I1_J0_1K] + f[L_1I_1J_K0] + f[L_I1_J1_K0] +
                         f[L_I1_1J_K0] + f[L_1I_J1_K0])

        double jz_neg = cr*(f[L_I0_J0_1K] + f[L_I0_1J_1K] + f[L_I0_J1_1K] +
                            f[L_1I_J0_1K] + f[L_I1_J0_1K])
        double den_bc = (cr*den_kn + jz_neg)/(cr - vel[Z])
        
        double jx = den_bc*inv_cr*vel[X]
        double jy = den_bc*inv_cr*vel[Y]
        double jz = den_bc*inv_cr*vel[Z]
                 
    _assign_f_1z(jx, jy, jz, f)
    
# ---------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
cdef void fix_vel_face_z1_D3Q19(double den, double *vel, double *f) nogil:
    """Enforce velocity at the boundary with outward unit normal = (0,0,1).
    The boundary condition scheme by Zou & He is applied.
    
    Parameters
    ----------
    den : double
          density, not used
    vel : double* (array, element count >= 3, read)
          velocity, enforced
    f : double* (array, element count >= 19, read and write)
        distribution values associated with the lattice vectors
        L_1I_J0_1K, L_I0_1J_1K, L_I0_J0_1K, L_I0_J1_1K, L_I1_J0_1K
        are assigned
    """
    cdef:
        double cr = 1.0, inv_cr = 1.0

        double den_kn = (f[L_I0_J0_K0] + f[L_1I_J0_K0] + f[L_I1_J0_K0] +
                         f[L_I0_1J_K0] + f[L_I0_J1_K0] + f[L_I0_J0_K1] +
                         f[L_I0_1J_K1] + f[L_I0_J1_K1] + f[L_1I_J0_K1] +
                         f[L_I1_J0_K1] + f[L_1I_1J_K0] + f[L_I1_J1_K0] +
                         f[L_I1_1J_K0] + f[L_1I_J1_K0])
      
        double jz_pos = cr*(f[L_I0_J0_K1] + f[L_I0_1J_K1] + f[L_I0_J1_K1] +
                            f[L_1I_J0_K1] + f[L_I1_J0_K1])
        double den_bc = (cr*den_kn + jz_pos)/(cr + vel[Z])
        
        double jx = den_bc*inv_cr*vel[X]
        double jy = den_bc*inv_cr*vel[Y]
        double jz = den_bc*inv_cr*vel[Z]
                 
    _assign_f_z1(jx, jy, jz, f)
    
# ---------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
cdef inline void _assign_f_1x(double jx, double jy, double jz,
                              double *f) nogil:
    """Assign distribution values, according to the Zou & He boundary scheme,
    at the boundary with outward unit normal = (-1,0,0).
    
    Parameters
    ----------
    jx : double
         x-component of the momentum density vector (divided by cr)
    jy : double
         y-component of the momentum density vector (divided by cr)
    jz : double
         z-component of the momentum density vector (divided by cr)
    f : double* (array, element count >= 19, read and write)
        distribution values associated with the lattice vectors
        L_I1_1J_K0, L_I1_J0_1K, L_I1_J0_K0, L_I1_J0_K1, L_I1_J1_K0
        are assigned
    """
    f[L_I1_1J_K0] = (f[L_1I_J1_K0] + (1.0/6.0)*jx + 0.5*(f[L_I0_J1_1K] + 
                     f[L_I0_J1_K0] + f[L_I0_J1_K1] - f[L_I0_1J_1K] -
                     f[L_I0_1J_K0] - f[L_I0_1J_K1]) - 0.5*jy)

    f[L_I1_J0_1K] = (f[L_1I_J0_K1] + (1.0/6.0)*jx + 0.5*(f[L_I0_1J_K1] +
                     f[L_I0_J0_K1] + f[L_I0_J1_K1] - f[L_I0_1J_1K] -
                     f[L_I0_J0_1K] - f[L_I0_J1_1K]) - 0.5*jz)
                     
    f[L_I1_J0_K0] = f[L_1I_J0_K0]  + (1.0/3.0)*jx
    
    f[L_I1_J0_K1] = (f[L_1I_J0_1K] + (1.0/6.0)*jx - 0.5*(f[L_I0_1J_K1] +
                     f[L_I0_J0_K1] + f[L_I0_J1_K1] - f[L_I0_1J_1K] -
                     f[L_I0_J0_1K] - f[L_I0_J1_1K]) + 0.5*jz)
                     
    f[L_I1_J1_K0] = (f[L_1I_1J_K0] + (1.0/6.0)*jx - 0.5*(f[L_I0_J1_1K] +
                     f[L_I0_J1_K0] + f[L_I0_J1_K1] - f[L_I0_1J_1K] -
                     f[L_I0_1J_K0] - f[L_I0_1J_K1]) + 0.5*jy)

# ---------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
cdef inline void _assign_f_x1(double jx, double jy, double jz,
                              double *f) nogil:
    """Assign distribution values, according to the Zou & He boundary scheme,
    at the boundary with outward unit normal = (1,0,0).
    
    Parameters
    ----------
    jx : double
         x-component of the momentum density vector (divided by cr)
    jy : double
         y-component of the momentum density vector (divided by cr)
    jz : double
         z-component of the momentum density vector (divided by cr)
    f : double* (array, element count >= 19, read and write)
        distribution values associated with the lattice vectors
        L_1I_1J_K0, L_1I_J0_1K, L_1I_J0_K0, L_1I_J0_K1, L_1I_J1_K0
        are assigned
    """
    f[L_1I_1J_K0] = (f[L_I1_J1_K0] - (1.0/6.0)*jx + 0.5*(f[L_I0_J1_1K] +
                     f[L_I0_J1_K0] + f[L_I0_J1_K1] - f[L_I0_1J_1K] -
                     f[L_I0_1J_K0] - f[L_I0_1J_K1]) - 0.5*jy)

    f[L_1I_J0_1K] = (f[L_I1_J0_K1] - (1.0/6.0)*jx + 0.5*(f[L_I0_1J_K1] +
                     f[L_I0_J0_K1] + f[L_I0_J1_K1] - f[L_I0_1J_1K] -
                     f[L_I0_J0_1K] - f[L_I0_J1_1K]) - 0.5*jz)

    f[L_1I_J0_K0] = f[L_I1_J0_K0]  - (1.0/3.0)*jx
 
    f[L_1I_J0_K1] = (f[L_I1_J0_1K] - (1.0/6.0)*jx - 0.5*(f[L_I0_1J_K1] +
                     f[L_I0_J0_K1] + f[L_I0_J1_K1] - f[L_I0_1J_1K] -
                     f[L_I0_J0_1K] - f[L_I0_J1_1K]) + 0.5*jz)

    f[L_1I_J1_K0] = (f[L_I1_1J_K0] - (1.0/6.0)*jx - 0.5*(f[L_I0_J1_1K] +
                     f[L_I0_J1_K0] + f[L_I0_J1_K1] - f[L_I0_1J_1K] -
                     f[L_I0_1J_K0] - f[L_I0_1J_K1]) + 0.5*jy)

# ---------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
cdef inline void _assign_f_1y(double jx, double jy, double jz,
                              double *f) nogil:
    """Assign distribution values, according to the Zou & He boundary scheme,
    at the boundary with outward unit normal = (0,-1,0).
    
    Parameters
    ----------
    jx : double
         x-component of the momentum density vector (divided by cr)
    jy : double
         y-component of the momentum density vector (divided by cr)
    jz : double
         z-component of the momentum density vector (divided by cr)
    f : double* (array, element count >= 19, read and write)
        distribution values associated with the lattice vectors
        L_1I_J1_K0, L_I0_J1_1K, L_I0_J1_K0, L_I0_J1_K1, L_I1_J1_K0
        are assigned
    """
    f[L_1I_J1_K0] = (f[L_I1_1J_K0] + (1.0/6.0)*jy + 0.5*(f[L_I1_J0_1K] +
                     f[L_I1_J0_K0] + f[L_I1_J0_K1] - f[L_1I_J0_1K] -
                     f[L_1I_J0_K0] - f[L_1I_J0_K1]) - 0.5*jx)

    f[L_I0_J1_1K] = (f[L_I0_1J_K1] + (1.0/6.0)*jy + 0.5*(f[L_I1_J0_K1] +
                     f[L_I0_J0_K1] + f[L_1I_J0_K1] - f[L_I1_J0_1K] -
                     f[L_I0_J0_1K] - f[L_1I_J0_1K]) - 0.5*jz)

    f[L_I0_J1_K0] = f[L_I0_1J_K0]  + (1.0/3.0)*jy

    f[L_I0_J1_K1] = (f[L_I0_1J_1K] + (1.0/6.0)*jy - 0.5*(f[L_I1_J0_K1] +
                     f[L_I0_J0_K1] + f[L_1I_J0_K1] - f[L_I1_J0_1K] -
                     f[L_I0_J0_1K] - f[L_1I_J0_1K]) + 0.5*jz)

    f[L_I1_J1_K0] = (f[L_1I_1J_K0] + (1.0/6.0)*jy - 0.5*(f[L_I1_J0_1K] +
                     f[L_I1_J0_K0] + f[L_I1_J0_K1] - f[L_1I_J0_1K] -
                     f[L_1I_J0_K0] - f[L_1I_J0_K1]) + 0.5*jx)
 
# ---------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
cdef inline void _assign_f_y1(double jx, double jy, double jz,
                              double *f) nogil:
    """Assign distribution values, according to the Zou & He boundary scheme,
    at the boundary with outward unit normal = (0,1,0).
    
    Parameters
    ----------
    jx : double
         x-component of the momentum density vector (divided by cr)
    jy : double
         y-component of the momentum density vector (divided by cr)
    jz : double
         z-component of the momentum density vector (divided by cr)
    f : double* (array, element count >= 19, read and write)
        distribution values associated with the lattice vectors
        L_1I_1J_K0, L_I0_1J_1K, L_I0_1J_K0, L_I0_1J_K1, L_I1_1J_K0
        are assigned
    """
    f[L_1I_1J_K0] = (f[L_I1_J1_K0] - (1.0/6.0)*jy + 0.5*(f[L_I1_J0_1K] +
                     f[L_I1_J0_K0] + f[L_I1_J0_K1] - f[L_1I_J0_1K] -
                     f[L_1I_J0_K0] - f[L_1I_J0_K1]) - 0.5*jx)

    f[L_I0_1J_1K] = (f[L_I0_J1_K1] - (1.0/6.0)*jy + 0.5*(f[L_I1_J0_K1] +
                     f[L_I0_J0_K1] + f[L_1I_J0_K1] - f[L_I1_J0_1K] -
                     f[L_I0_J0_1K] - f[L_1I_J0_1K]) - 0.5*jz)

    f[L_I0_1J_K0] = f[L_I0_J1_K0]  - (1.0/3.0)*jy

    f[L_I0_1J_K1] = (f[L_I0_J1_1K] - (1.0/6.0)*jy - 0.5*(f[L_I1_J0_K1] +
                     f[L_I0_J0_K1] + f[L_1I_J0_K1] - f[L_I1_J0_1K] - 
                     f[L_I0_J0_1K] - f[L_1I_J0_1K]) + 0.5*jz)

    f[L_I1_1J_K0] = (f[L_1I_J1_K0] - (1.0/6.0)*jy - 0.5*(f[L_I1_J0_1K] +
                     f[L_I1_J0_K0] + f[L_I1_J0_K1] - f[L_1I_J0_1K] -
                     f[L_1I_J0_K0] - f[L_1I_J0_K1]) + 0.5*jx)

# ---------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
cdef inline void _assign_f_1z(double jx, double jy, double jz,
                              double *f) nogil:
    """Assign distribution values, according to the Zou & He boundary scheme,
    at the boundary with outward unit normal = (0,0,-1).
    
    Parameters
    ----------
    jx : double
         x-component of the momentum density vector (divided by cr)
    jy : double
         y-component of the momentum density vector (divided by cr)
    jz : double
         z-component of the momentum density vector (divided by cr)
    f : double* (array, element count >= 19, read and write)
        distribution values associated with the lattice vectors
        L_1I_J0_K1, L_I0_1J_K1, L_I0_J0_K1, L_I0_J1_K1, L_I1_J0_K1
        are assigned
    """
    f[L_1I_J0_K1] = (f[L_I1_J0_1K] + (1.0/6.0)*jz + 0.5*(f[L_I1_J1_K0] +
                     f[L_I1_J0_K0] + f[L_I1_1J_K0] - f[L_1I_J1_K0] -
                     f[L_1I_J0_K0] - f[L_1I_1J_K0]) - 0.5*jx)
                                           
    f[L_I0_1J_K1] = (f[L_I0_J1_1K] + (1.0/6.0)*jz + 0.5*(f[L_I1_J1_K0] +
                     f[L_I0_J1_K0] + f[L_1I_J1_K0] - f[L_I1_1J_K0] -
                     f[L_I0_1J_K0] - f[L_1I_1J_K0]) - 0.5*jy)
                                           
    f[L_I0_J0_K1] = f[L_I0_J0_1K]  + (1.0/3.0)*jz
    
    f[L_I0_J1_K1] = (f[L_I0_1J_1K] + (1.0/6.0)*jz - 0.5*(f[L_I1_J1_K0] +
                     f[L_I0_J1_K0] + f[L_1I_J1_K0] - f[L_I1_1J_K0] -
                     f[L_I0_1J_K0] - f[L_1I_1J_K0]) + 0.5*jy)
                                           
    f[L_I1_J0_K1] = (f[L_1I_J0_1K] + (1.0/6.0)*jz - 0.5*(f[L_I1_J1_K0] +
                     f[L_I1_J0_K0] + f[L_I1_1J_K0] - f[L_1I_J1_K0] -
                     f[L_1I_J0_K0] - f[L_1I_1J_K0]) + 0.5*jx)

# ---------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
cdef inline void _assign_f_z1(double jx, double jy, double jz,
                              double *f) nogil:
    """Assign distribution values, according to the Zou & He boundary scheme,
    at the boundary with outward unit normal = (0,0,1).
    
    Parameters
    ----------
    jx : double
         x-component of the momentum density vector (divided by cr)
    jy : double
         y-component of the momentum density vector (divided by cr)
    jz : double
         z-component of the momentum density vector (divided by cr)
    f : double* (array, element count >= 19, read and write)
        distribution values associated with the lattice vectors
        L_1I_J0_1K, L_I0_1J_1K, L_I0_J0_1K, L_I0_J1_1K, L_I1_J0_1K
        are assigned
    """
    f[L_1I_J0_1K] = (f[L_I1_J0_K1] - (1.0/6.0)*jz + 0.5*(f[L_I1_J1_K0] +
                     f[L_I1_J0_K0] + f[L_I1_1J_K0] - f[L_1I_J1_K0] -
                     f[L_1I_J0_K0] - f[L_1I_1J_K0]) - 0.5*jx)
                                           
    f[L_I0_1J_1K] = (f[L_I0_J1_K1] - (1.0/6.0)*jz + 0.5*(f[L_I1_J1_K0] +
                     f[L_I0_J1_K0] + f[L_1I_J1_K0] - f[L_I1_1J_K0] -
                     f[L_I0_1J_K0] - f[L_1I_1J_K0]) - 0.5*jy)
                                           
    f[L_I0_J0_1K] = f[L_I0_J0_K1]  - (1.0/3.0)*jz
    
    f[L_I0_J1_1K] = (f[L_I0_1J_K1] - (1.0/6.0)*jz - 0.5*(f[L_I1_J1_K0] +
                     f[L_I0_J1_K0] + f[L_1I_J1_K0] - f[L_I1_1J_K0] -
                     f[L_I0_1J_K0] - f[L_1I_1J_K0]) + 0.5*jy)
                                           
    f[L_I1_J0_1K] = (f[L_1I_J0_K1] - (1.0/6.0)*jz - 0.5*(f[L_I1_J1_K0] +
                     f[L_I1_J0_K0] + f[L_I1_1J_K0] - f[L_1I_J1_K0] -
                     f[L_1I_J0_K0] - f[L_1I_1J_K0]) + 0.5*jx)

# ---------------------------------------------------------------------------
