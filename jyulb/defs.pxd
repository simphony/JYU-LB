"""Enumerations related to the coordinate system, geometry, and 
configuration of the fluid flow simulations.

Author
------
Keijo Mattila, JYU, March 2017.
"""
# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
import cython
cimport cython

# ---------------------------------------------------------------------------
# Enumeration of cartesian coordinate axes
# ---------------------------------------------------------------------------
cpdef enum:
    X = 0
    Y = 1
    Z = 2

# ---------------------------------------------------------------------------
# Enumeration of cuboid faces (cuboid aligned with the cart.coord.axes)
# Notation: 1X denotes the face with outward unit normal = (-1,0,0),
#           Y1 denotes the face with outward unit normal = (0,1,0), etc.
# ---------------------------------------------------------------------------
cpdef enum:
    FACE_1X = 0
    FACE_X1 = 1
    FACE_1Y = 2
    FACE_Y1 = 3
    FACE_1Z = 4
    FACE_Z1 = 5

# ---------------------------------------------------------------------------
# Enumeration of cuboid edges (cuboid aligned with the cart.coord.axes)
# Notation: 1X_1Y denotes the edge shared by the faces 1X and 1Y, etc.
# ---------------------------------------------------------------------------
cpdef enum:
    EDGE_1X_1Y = 0
    EDGE_1X_Y1 = 1
    EDGE_X1_1Y = 2
    EDGE_X1_Y1 = 3
    EDGE_1X_1Z = 4
    EDGE_1X_Z1 = 5
    EDGE_X1_1Z = 6
    EDGE_X1_Z1 = 7
    EDGE_1Y_1Z = 8
    EDGE_1Y_Z1 = 9
    EDGE_Y1_1Z = 10
    EDGE_Y1_Z1 = 11
    
# ---------------------------------------------------------------------------
# Enumeration of cuboid corners (cuboid aligned with the cart.coord.axes)
# Notation: 1X_1Y_1Z denotes the corner shared by the faces 1X, 1Y, and 1Z,
#           etc.
# ---------------------------------------------------------------------------
cpdef enum:
    CORNER_1X_1Y_1Z = 0
    CORNER_1X_1Y_Z1 = 1
    CORNER_1X_Y1_1Z = 2
    CORNER_1X_Y1_Z1 = 3
    CORNER_X1_1Y_1Z = 4
    CORNER_X1_1Y_Z1 = 5
    CORNER_X1_Y1_1Z = 6
    CORNER_X1_Y1_Z1 = 7

# ---------------------------------------------------------------------------
# Enumeration of phases
# ---------------------------------------------------------------------------
cpdef enum:
    SOLID  = 0
    FLUID  = 1
    LIQUID = 2
    VAPOUR = 3
    
# ---------------------------------------------------------------------------
# Enumeration of flow regimes
# ---------------------------------------------------------------------------
cpdef enum:
    STOKES    = 0
    LAMINAR   = 1
    TURBULENT = 2

# ---------------------------------------------------------------------------
# Enumeration of collision operators
# ---------------------------------------------------------------------------
cpdef enum:
    BGK = 0
    TRT = 1
    REGULARIZATION = 2
    RECURRENCE_REL = 3

# ---------------------------------------------------------------------------
# Enumeration of boundary conditions
# ---------------------------------------------------------------------------
cpdef enum:
    WALL          = 0 # immobile and adiabatic
    PERIODIC      = 1
    FIXED_DEN     = 2 # note: defines implicitly the boundary normal velocity
    FIXED_VEL     = 3 # note: defines implicitly the boundary density 
    SYMMETRIC     = 4 # aka specular reflection
#    ANTISYMMETRIC = 5 # aka inverse reflection
# ---------------------------------------------------------------------------
    