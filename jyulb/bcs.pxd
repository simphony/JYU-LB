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

# ---------------------------------------------------------------------------
# Data type definitions (pointer to a bc enforcing function)
# ---------------------------------------------------------------------------
ctypedef void (*bcs_fpntr)(double, double*, double*) nogil

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
cdef void fix_den_face_1x_D3Q19(double den, double *vel, double *f) nogil
cdef void fix_den_face_x1_D3Q19(double den, double *vel, double *f) nogil
cdef void fix_den_face_1y_D3Q19(double den, double *vel, double *f) nogil
cdef void fix_den_face_y1_D3Q19(double den, double *vel, double *f) nogil
cdef void fix_den_face_1z_D3Q19(double den, double *vel, double *f) nogil
cdef void fix_den_face_z1_D3Q19(double den, double *vel, double *f) nogil

cdef void fix_vel_face_1x_D3Q19(double den, double *vel, double *f) nogil
cdef void fix_vel_face_x1_D3Q19(double den, double *vel, double *f) nogil
cdef void fix_vel_face_1y_D3Q19(double den, double *vel, double *f) nogil
cdef void fix_vel_face_y1_D3Q19(double den, double *vel, double *f) nogil
cdef void fix_vel_face_1z_D3Q19(double den, double *vel, double *f) nogil
cdef void fix_vel_face_z1_D3Q19(double den, double *vel, double *f) nogil

cdef void _assign_f_1x(double jx, double jy, double jz, double *f) nogil
cdef void _assign_f_x1(double jx, double jy, double jz, double *f) nogil
cdef void _assign_f_1y(double jx, double jy, double jz, double *f) nogil
cdef void _assign_f_y1(double jx, double jy, double jz, double *f) nogil
cdef void _assign_f_1z(double jx, double jy, double jz, double *f) nogil
cdef void _assign_f_z1(double jx, double jy, double jz, double *f) nogil
# ---------------------------------------------------------------------------
