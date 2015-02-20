//===========================================================================
// Author: Keijo Mattila, JYU, February 2015
// Description: Input and output of lattice data in raw format
// Details: 
//===========================================================================
#ifndef LB_RAW_IO_H
#define LB_RAW_IO_H
//===========================================================================
#include "lattice.h"
//===========================================================================
namespace LB_Raw_IO
{
  extern bool read_geom(const char *base_fname, Lattice::SparseGeom3D *lat);
  extern bool write_geom(const char *base_fname, Lattice::SparseGeom3D *lat);
                 
  extern bool read_den(const char *base_fname, Lattice::Isothermal3D *lat);
  extern bool write_den(const char *base_fname, Lattice::Isothermal3D *lat);

  extern bool read_vel(const char *base_fname, Lattice::Isothermal3D *lat);
  extern bool write_vel(const char *base_fname, Lattice::Isothermal3D *lat);

  extern bool read_frc(const char *base_fname, Lattice::Isothermal3D *lat);
  extern bool write_frc(const char *base_fname, Lattice::Isothermal3D *lat);
}
//===========================================================================
#endif
//===========================================================================