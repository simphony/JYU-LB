"""Analytical expressions for particular flow fields.

Author
------
Keijo Mattila, JYU, April 2017.
"""
# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
import cython
cimport cython
from cython cimport boundscheck, wraparound, cdivision

# ---------------------------------------------------------------------------
# Flow fields
# ---------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef void poise_plate_steady_state_pgrad(int H, double dvisc, double pgrad,
                                          double [:] centrel_dist,
                                          double [:] ana_vel):
    """Steady-state Poiseuille flow (i.e. flow between two parallel plates).
    Plates are located half a lattice spacing away from the lowermost and
    uppermost lattice site.
    
    Parameters
    ----------
    H : int
        height of the channel in lattice spacings
    dvisc : double
            dynamic viscosity
    pgrad : double
             pressure gradient
    centrel_dist : double[:] (typed memoryview, element count >= H, write)
                   distance of the lattice sites from the centreline
    ana_vel : double[:] (typed memoryview, element count >= H, write)
              analytical flow velocity at the lattice sites
              (only the longitudal velocity component is returned)
    """
    cdef:
        int x
        double channel_h = 0.5*H, rel_dist
        double umax = -0.5*channel_h*channel_h*pgrad/dvisc

    for x in range(H):
        centrel_dist[x] = (0.5 + x) - channel_h
        rel_dist = centrel_dist[x]/channel_h
        ana_vel[x] = umax*(1.0 - rel_dist*rel_dist)

# ---------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef void poise_plate_steady_state_umax(int H, double umax,
                                         double [:] centrel_dist,
                                         double [:] ana_vel):
    """Steady-state Poiseuille flow (i.e. flow between two parallel plates).
    Plates are located half a lattice spacing away from the lowermost and
    uppermost lattice site.
    
    Parameters
    ----------
    H : int
        height of the channel in lattice spacings
    umax : double
           maximum velocity (at the centerline)
    centrel_dist : double[:] (typed memoryview, element count >= H, write)
                   distance of the lattice sites from the centreline
    ana_vel : double[:] (typed memoryview, element count >= H, write)
              analytical flow velocity at the lattice sites
              (only the longitudal velocity component is returned)
    """
    cdef:
        int x
        double channel_h = 0.5*H, rel_dist

    for x in range(H):
        centrel_dist[x] = (0.5 + x) - channel_h
        rel_dist = centrel_dist[x]/channel_h
        ana_vel[x] = umax*(1.0 - rel_dist*rel_dist)
        
# ---------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef void couette_plate_steady_state(int H, double low_v1, double low_v2,
                                      double upp_v1, double upp_v2,
                                      double [:] centrel_dist,
                                      double [:,::1] ana_vel):
    """Steady-state Couette flow (i.e. flow between two moving, parallel
    plates). Plates are located half a lattice spacing away from
    the lowermost and uppermost lattice site.
    
    Parameters
    ----------
    H : int
        height of the channel in lattice spacings
    low_v1 : double
             velocity of the lower plate in the tangential direction 1
    low_v2 : double
             velocity of the lower plate in the tangential direction 2
    upp_v1 : double
             velocity of the upper plate in the tangential direction 1
    upp_v2 : double
             velocity of the upper plate in the tangential direction 2
    centrel_dist : double[:] (typed memoryview, element count >= H, write)
                   distance of the lattice sites from the centreline
    ana_vel : double[:,::1] (typed memoryview, element count (>= H,2), write)
              analytical flow velocity at the lattice sites
              (vel.components returned only in the tangential directions)
    """
    cdef:
        int x
        double channel_h = 0.5*H, dist
        double grad_v1 = (upp_v1 - low_v1)/float(H)
        double grad_v2 = (upp_v2 - low_v2)/float(H)

    for x in range(H):
        dist_from_low = 0.5 + x
        centrel_dist[x] = dist_from_low - channel_h
        ana_vel[x,0] = low_v1 + dist_from_low*grad_v1
        ana_vel[x,1] = low_v2 + dist_from_low*grad_v2

# ---------------------------------------------------------------------------
