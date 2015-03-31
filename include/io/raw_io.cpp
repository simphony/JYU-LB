//===========================================================================
// Author: Keijo Mattila, JYU, February 2015
// Description: Input and output of lattice data in raw format
// Details: 
//===========================================================================
#include <omp.h>
#include <cstring>
#include <stdio.h>
#include "raw_io.h"
//===========================================================================
IsothermalRawIO::IsothermalRawIO(Lattice *lat)
{
  UINT size[3];
  lat->get_size(size);
  _ncount = lat->get_node_count();
  _nx = size[0]; _ny = size[1]; _nz = size[2];
}
//---------------------------------------------------------------------------
IsothermalRawIO::~IsothermalRawIO()
{
}
//---------------------------------------------------------------------------
bool IsothermalRawIO::read_geom(const char *bfname, GeomData *phase)
{
  FILE *ifile;
  if( !open_file(bfname, ".geom.in.raw", "rb", &ifile) ) return false;
  
  unsigned char *read_buff = new unsigned char[_ncount];
  fread(read_buff, sizeof(unsigned char), _ncount, ifile);
    
  #pragma omp parallel for
  for(unsigned int k = 0; k < _nz; ++k) {
    for(unsigned int j = 0; j < _ny; ++j) {
      for(unsigned int i = 0; i < _nx; ++i)
      {
        UINT ijk[3] = {i,j,k}, raw_n = raw_ijk2n(i,j,k);
        phase->set_val_ijk(ijk, read_buff[raw_n]);
      }
    }
  }
  delete [] read_buff;
  fclose(ifile);
  return true;
}
//---------------------------------------------------------------------------
bool IsothermalRawIO::write_geom(const char *bfname, GeomData *phase)
{
  FILE *ofile;
  if( !open_file(bfname, ".geom.out.raw", "wb", &ofile) ) return false;
  
  unsigned char *write_buff = new unsigned char[_ncount];
    
  #pragma omp parallel for
  for(unsigned int k = 0; k < _nz; ++k) {
    for(unsigned int j = 0; j < _ny; ++j) {
      for(unsigned int i = 0; i < _nx; ++i)
      {
        UINT ijk[3] = {i,j,k}, raw_n = raw_ijk2n(i,j,k);
        write_buff[raw_n] = phase->get_val_ijk(ijk); 
      }
    }
  }
  fwrite(write_buff, sizeof(unsigned char), _ncount, ofile);

  delete [] write_buff;;
  fclose(ofile);
  return true;
}
//---------------------------------------------------------------------------
bool IsothermalRawIO::read_den(const char *bfname, IsothermalNodeData *data)
{
  FILE *ifile;
  if( !open_file(bfname, ".den.in.raw", "rb", &ifile) ) return false;
  
  FieldData *den = data->den();
  NodeSet *fnodes = data->get_nodeset();
  UINT fluid_ncount = fnodes->get_node_count();

  double *read_buff = new double[_ncount];
  fread(read_buff, sizeof(double), _ncount, ifile);
    
  #pragma omp parallel for
  for(UINT fnode_enum = 0; fnode_enum < fluid_ncount; ++fnode_enum)
  {
    UINT ijk[3];
    fnodes->get_ijk(fnode_enum, ijk);

    UINT raw_n = raw_ijk2n(ijk[0],ijk[1],ijk[2]);
    den->set_val_n(fnode_enum, read_buff[raw_n]);
  }
  delete [] read_buff;
  fclose(ifile);
  return true;
}
//---------------------------------------------------------------------------
bool IsothermalRawIO::write_den(const char *bfname, IsothermalNodeData *data)
{
  FILE *ofile;
  if( !open_file(bfname, ".den.out.raw", "wb", &ofile) ) return false;
  
  FieldData *den = data->den();
  NodeSet *fnodes = data->get_nodeset();
  UINT fluid_ncount = fnodes->get_node_count();

  double iden = data->get_iden(), *write_buff = new double[_ncount];
  #pragma omp parallel for
  for(UINT n = 0; n < _ncount; ++n) {write_buff[n] = iden;}
    
  #pragma omp parallel for
  for(UINT fnode_enum = 0; fnode_enum < fluid_ncount; ++fnode_enum)
  {
    UINT ijk[3];
    fnodes->get_ijk(fnode_enum, ijk);

    UINT raw_n = raw_ijk2n(ijk[0],ijk[1],ijk[2]);
    write_buff[raw_n] = den->get_val_n(fnode_enum); 
  }
  fwrite(write_buff, sizeof(double), _ncount, ofile);

  delete [] write_buff;;
  fclose(ofile);
  return true;
}
//---------------------------------------------------------------------------
bool IsothermalRawIO::read_vel(const char *bfname, IsothermalNodeData *data)
{
  FILE *ifile;
  if( !open_file(bfname, ".vel.in.raw", "rb", &ifile) ) return false;
  
  FieldData *vx = data->velx(), *vy = data->vely(), *vz = data->velz();
  NodeSet *fnodes = data->get_nodeset();
  UINT fluid_ncount = fnodes->get_node_count();

  double *read_buff = new double[3*_ncount];
  fread(read_buff, sizeof(double), 3*_ncount, ifile);
    
  #pragma omp parallel for
  for(UINT fnode_enum = 0; fnode_enum < fluid_ncount; ++fnode_enum)
  {
    UINT ijk[3];
    fnodes->get_ijk(fnode_enum, ijk);
    vx->set_val_n(fnode_enum, read_buff[raw_ijkl2n(ijk[0],ijk[1],ijk[2],0)]);
    vy->set_val_n(fnode_enum, read_buff[raw_ijkl2n(ijk[0],ijk[1],ijk[2],1)]);
    vz->set_val_n(fnode_enum, read_buff[raw_ijkl2n(ijk[0],ijk[1],ijk[2],2)]);
  }
  delete [] read_buff;
  fclose(ifile);
  return true;
}
//---------------------------------------------------------------------------
bool IsothermalRawIO::write_vel(const char *bfname, IsothermalNodeData *data)
{
  FILE *ofile;
  if( !open_file(bfname, ".vel.out.raw", "wb", &ofile) ) return false;
  
  FieldData *vx = data->velx(), *vy = data->vely(), *vz = data->velz();
  NodeSet *fnodes = data->get_nodeset();
  UINT fluid_ncount = fnodes->get_node_count();

  double ivel[3], *write_buff = new double[3*_ncount];
  data->get_ivel(ivel);
  
  #pragma omp parallel for
  for(UINT n = 0; n < _ncount; ++n) {
    write_buff[3*n + 0] = ivel[0];
    write_buff[3*n + 1] = ivel[1];
    write_buff[3*n + 2] = ivel[2];
  }
    
  #pragma omp parallel for
  for(UINT fnode_enum = 0; fnode_enum < fluid_ncount; ++fnode_enum)
  {
    UINT ijk[3];
    fnodes->get_ijk(fnode_enum, ijk);
    write_buff[raw_ijkl2n(ijk[0],ijk[1],ijk[2],0)] = vx->get_val_n(fnode_enum);
    write_buff[raw_ijkl2n(ijk[0],ijk[1],ijk[2],1)] = vy->get_val_n(fnode_enum);
    write_buff[raw_ijkl2n(ijk[0],ijk[1],ijk[2],2)] = vz->get_val_n(fnode_enum);
  }
  fwrite(write_buff, sizeof(double), 3*_ncount, ofile);

  delete [] write_buff;;
  fclose(ofile);
  return true;
}
//---------------------------------------------------------------------------
bool IsothermalRawIO::read_frc(const char *bfname, IsothermalNodeData *data)
{
  FILE *ifile;
  if( !open_file(bfname, ".force.in.raw", "rb", &ifile) ) return false;
  
  FieldData *fx = data->frcx(), *fy = data->frcy(), *fz = data->frcz();
  NodeSet *fnodes = data->get_nodeset();
  UINT fluid_ncount = fnodes->get_node_count();

  double *read_buff = new double[3*_ncount];
  fread(read_buff, sizeof(double), 3*_ncount, ifile);
    
  #pragma omp parallel for
  for(UINT fnode_enum = 0; fnode_enum < fluid_ncount; ++fnode_enum)
  {
    UINT ijk[3];
    fnodes->get_ijk(fnode_enum, ijk);
    fx->set_val_n(fnode_enum, read_buff[raw_ijkl2n(ijk[0],ijk[1],ijk[2],0)]);
    fy->set_val_n(fnode_enum, read_buff[raw_ijkl2n(ijk[0],ijk[1],ijk[2],1)]);
    fz->set_val_n(fnode_enum, read_buff[raw_ijkl2n(ijk[0],ijk[1],ijk[2],2)]);
  }
  delete [] read_buff;
  fclose(ifile);
  return true;
}
//---------------------------------------------------------------------------
bool IsothermalRawIO::write_frc(const char *bfname, IsothermalNodeData *data)
{
  FILE *ofile;
  if( !open_file(bfname, ".force.out.raw", "wb", &ofile) ) return false;
  
  FieldData *fx = data->frcx(), *fy = data->frcy(), *fz = data->frcz();
  NodeSet *fnodes = data->get_nodeset();
  UINT fluid_ncount = fnodes->get_node_count();

  double ifrc[3], *write_buff = new double[3*_ncount];
  data->get_ifrc(ifrc);
  
  #pragma omp parallel for
  for(UINT n = 0; n < _ncount; ++n) {
    write_buff[3*n + 0] = ifrc[0];
    write_buff[3*n + 1] = ifrc[1];
    write_buff[3*n + 2] = ifrc[2];
  }
    
  #pragma omp parallel for
  for(UINT fnode_enum = 0; fnode_enum < fluid_ncount; ++fnode_enum)
  {
    UINT ijk[3];
    fnodes->get_ijk(fnode_enum, ijk);
    write_buff[raw_ijkl2n(ijk[0],ijk[1],ijk[2],0)] = fx->get_val_n(fnode_enum);
    write_buff[raw_ijkl2n(ijk[0],ijk[1],ijk[2],1)] = fy->get_val_n(fnode_enum);
    write_buff[raw_ijkl2n(ijk[0],ijk[1],ijk[2],2)] = fz->get_val_n(fnode_enum);
  }
  fwrite(write_buff, sizeof(double), 3*_ncount, ofile);

  delete [] write_buff;;
  fclose(ofile);
  return true;
}
//---------------------------------------------------------------------------
UINT IsothermalRawIO::raw_ij2n(UINT i, UINT j)
{
  return j*_nx + i;
}
//---------------------------------------------------------------------------
UINT IsothermalRawIO::raw_ijk2n(UINT i, UINT j, UINT k)
{
  return k*_nx*_ny + j*_nx + i;
}
//---------------------------------------------------------------------------
UINT IsothermalRawIO::raw_ijkl2n(UINT i, UINT j, UINT k, UINT l)
{
  return k*3*_nx*_ny + j*3*_nx + i*3 + l;
}
//---------------------------------------------------------------------------
bool IsothermalRawIO::open_file(const char *bfname, const char *ext,
                           const char *mode, FILE **file)
{
  char fname[256];
  strcpy(fname, bfname);
  strcat(fname, ext);

  (*file) = fopen(fname, mode);
  if( (*file) == 0 ) {
    printf("Warning: cannot open file %s!\n", fname);
    return false;
  }
  return true;
}
//===========================================================================