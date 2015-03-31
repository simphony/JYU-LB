//===========================================================================
// Author: Keijo Mattila, JYU, February 2015
// Description: Input and output of lattice data in raw format
// Details: 
//===========================================================================
#ifndef RAW_IO_H
#define RAW_IO_H
//===========================================================================
#include "data.h"
//===========================================================================
class IsothermalRawIO
{
  public:
    IsothermalRawIO(Lattice *lat);
    ~IsothermalRawIO();
    
    bool read_geom(const char *bfname, GeomData *phase);
    bool write_geom(const char *bfname, GeomData *phase);
                 
    bool read_den(const char *bfname, IsothermalNodeData *data);
    bool write_den(const char *bfname, IsothermalNodeData *data);

    bool read_vel(const char *bfname, IsothermalNodeData *data);
    bool write_vel(const char *bfname, IsothermalNodeData *data);

    bool read_frc(const char *bfname, IsothermalNodeData *data);
    bool write_frc(const char *bfname, IsothermalNodeData *data);

  private:
    UINT _ncount, _nx, _ny, _nz;

    UINT raw_ij2n(UINT i, UINT j);
    UINT raw_ijk2n(UINT i, UINT j, UINT k);
    UINT raw_ijkl2n(UINT i, UINT j, UINT k, UINT l);

    bool open_file(const char *bfname, const char *ext, const char *mode,
                   FILE **file);
};
//===========================================================================
#endif
//===========================================================================