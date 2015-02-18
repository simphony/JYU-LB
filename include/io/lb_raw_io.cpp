//===========================================================================
// Author: Keijo Mattila, JYU, February 2015
// Description: Input and output of lattice data in raw format
// Details: 
//===========================================================================
#include <omp.h>
#include <cstring>
#include <stdio.h>
#include "lb_raw_io.h"
//===========================================================================
unsigned int RAW_IJ_TO_N(unsigned int nx, unsigned int ny,
                         unsigned int i, unsigned int j)
{
  return j*nx + i;
}
//---------------------------------------------------------------------------
unsigned int RAW_IJK_TO_N(unsigned int nx, unsigned int ny, unsigned int nz,
                          unsigned int i, unsigned int j, unsigned int k)
{
  return k*nx*ny + j*nx + i;
}
//---------------------------------------------------------------------------
unsigned int RAW_IJKL_TO_N(unsigned int nx, unsigned int ny, unsigned int nz,
                           unsigned int i, unsigned int j, unsigned int k,
                           unsigned int l)
{
  return k*3*nx*ny + j*3*nx + i*3 + l;
}
//---------------------------------------------------------------------------
bool open_file(const char *base_fname, const char *ext, const char *mode,
               FILE **file)
{
  char fname[256];
  strcpy(fname, base_fname);
  strcat(fname, ext);

  (*file) = fopen(fname, mode);
  if( (*file) == 0 ) {
    printf("Warning: cannot open file %s!\n", fname);
    return false;
  }
  return true;
}
//---------------------------------------------------------------------------
bool LB_Raw_IO::read_geom(const char *base_fname, Lattice::SparseGeom3D *lat)
{
  FILE *ifile;
  if( !open_file(base_fname, ".geom.in.raw", "rb", &ifile) ) return false;
  
  unsigned int nx = lat->nx(), ny = lat->ny(),
    nz = lat->nz(), ncount = nx*ny*nz;
  unsigned char *read_buff = new unsigned char[ncount],
    *geom = new unsigned char[ncount];

  fread(read_buff, sizeof(unsigned char), ncount, ifile);
    
  #pragma omp parallel for
  for(unsigned int k = 0; k < nz; ++k) {
    for(unsigned int j = 0; j < ny; ++j) {
      for(unsigned int i = 0; i < nx; ++i)
      {
        unsigned int raw_n = RAW_IJK_TO_N(nx, ny, nz, i, j, k);
        geom[lat->IJK_TO_N(i,j,k)] = read_buff[raw_n];
      }
    }
  }
  lat->set_geom(geom);
  
  delete [] read_buff; delete [] geom;
  fclose(ifile);
  return true;
}
//---------------------------------------------------------------------------
bool LB_Raw_IO::write_geom(const char *base_fname,Lattice::SparseGeom3D *lat)
{
  FILE *ofile;
  if( !open_file(base_fname, ".geom.out.raw", "wb", &ofile) ) return false;
  
  unsigned int nx = lat->nx(), ny = lat->ny(),
    nz = lat->nz(), ncount = nx*ny*nz;
  unsigned char *write_buff = new unsigned char[ncount];
    
  #pragma omp parallel for
  for(unsigned int k = 0; k < nz; ++k) {
    for(unsigned int j = 0; j < ny; ++j) {
      for(unsigned int i = 0; i < nx; ++i)
      {
        unsigned int raw_n = RAW_IJK_TO_N(nx, ny, nz, i, j, k);
        write_buff[raw_n] = lat->get_geom_ijk(i,j,k); 
      }
    }
  }
  fwrite(write_buff, sizeof(unsigned char), ncount, ofile);

  delete [] write_buff;;
  fclose(ofile);
  return true;
}
//---------------------------------------------------------------------------
bool LB_Raw_IO::read_den(const char *base_fname, Lattice::Isothermal3D *lat)
{
  FILE *ifile;
  if( !open_file(base_fname, ".den.in.raw", "rb", &ifile) ) return false;
  
  unsigned int nx = lat->nx(), ny = lat->ny(),
    nz = lat->nz(), ncount = nx*ny*nz;
  double *read_buff = new double[ncount];

  fread(read_buff, sizeof(double), ncount, ifile);
    
  #pragma omp parallel for
  for(unsigned int k = 0; k < nz; ++k) {
    for(unsigned int j = 0; j < ny; ++j) {
      for(unsigned int i = 0; i < nx; ++i)
      {
        unsigned int raw_n = RAW_IJK_TO_N(nx, ny, nz, i, j, k);
        lat->set_den_ijk(i, j, k, read_buff[raw_n]);
      }
    }
  }
  delete [] read_buff;
  fclose(ifile);
  return true;
}
//---------------------------------------------------------------------------
bool LB_Raw_IO::write_den(const char *base_fname, Lattice::Isothermal3D *lat)
{
  FILE *ofile;
  if( !open_file(base_fname, ".den.out.raw", "wb", &ofile) ) return false;
  
  unsigned int nx = lat->nx(), ny = lat->ny(),
    nz = lat->nz(), ncount = nx*ny*nz;
  double *write_buff = new double[ncount];
    
  #pragma omp parallel for
  for(unsigned int k = 0; k < nz; ++k) {
    for(unsigned int j = 0; j < ny; ++j) {
      for(unsigned int i = 0; i < nx; ++i)
      {
        double den;
        lat->get_den_ijk(i,j,k,&den);
        write_buff[RAW_IJK_TO_N(nx, ny, nz, i, j, k)] = den; 
      }
    }
  }
  fwrite(write_buff, sizeof(double), ncount, ofile);

  delete [] write_buff;;
  fclose(ofile);
  return true;
}
//---------------------------------------------------------------------------
bool LB_Raw_IO::read_vel(const char *base_fname, Lattice::Isothermal3D *lat)
{
  FILE *ifile;
  if( !open_file(base_fname, ".vel.in.raw", "rb", &ifile) ) return false;
  
  unsigned int nx = lat->nx(), ny = lat->ny(),
    nz = lat->nz(), ncount = 3*nx*ny*nz;
  double *read_buff = new double[ncount];

  fread(read_buff, sizeof(double), ncount, ifile);
    
  #pragma omp parallel for
  for(unsigned int k = 0; k < nz; ++k) {
    for(unsigned int j = 0; j < ny; ++j) {
      for(unsigned int i = 0; i < nx; ++i)
      {
        double ux = read_buff[RAW_IJKL_TO_N(nx,ny,nz,i,j,k,0)],
          uy = read_buff[RAW_IJKL_TO_N(nx,ny,nz,i,j,k,1)],
          uz = read_buff[RAW_IJKL_TO_N(nx,ny,nz,i,j,k,2)];

        lat->set_vel_ijk(i, j, k, ux, uy, uz);
      }
    }
  }
  delete [] read_buff;
  fclose(ifile);
  return true;
}
//---------------------------------------------------------------------------
bool LB_Raw_IO::write_vel(const char *base_fname, Lattice::Isothermal3D *lat)
{
  FILE *ofile;
  if( !open_file(base_fname, ".vel.out.raw", "wb", &ofile) ) return false;
  
  unsigned int nx = lat->nx(), ny = lat->ny(),
    nz = lat->nz(), ncount = 3*nx*ny*nz;
  double *write_buff = new double[ncount];
    
  #pragma omp parallel for
  for(unsigned int k = 0; k < nz; ++k) {
    for(unsigned int j = 0; j < ny; ++j) {
      for(unsigned int i = 0; i < nx; ++i)
      {
        double ux,uy,uz;
        lat->get_vel_ijk(i,j,k,&ux,&uy,&uz);
        write_buff[RAW_IJKL_TO_N(nx, ny, nz, i, j, k, 0)] = ux; 
        write_buff[RAW_IJKL_TO_N(nx, ny, nz, i, j, k, 1)] = uy; 
        write_buff[RAW_IJKL_TO_N(nx, ny, nz, i, j, k, 2)] = uz; 
      }
    }
  }
  fwrite(write_buff, sizeof(double), ncount, ofile);

  delete [] write_buff;;
  fclose(ofile);
  return true;
}
//---------------------------------------------------------------------------
bool LB_Raw_IO::read_frc(const char *base_fname, Lattice::Isothermal3D *lat)
{
  FILE *ifile;
  if( !open_file(base_fname, ".force.in.raw", "rb", &ifile) ) return false;
  
  unsigned int nx = lat->nx(), ny = lat->ny(),
    nz = lat->nz(), ncount = 3*nx*ny*nz;
  double *read_buff = new double[ncount];

  fread(read_buff, sizeof(double), ncount, ifile);
    
  #pragma omp parallel for
  for(unsigned int k = 0; k < nz; ++k) {
    for(unsigned int j = 0; j < ny; ++j) {
      for(unsigned int i = 0; i < nx; ++i)
      {
        double fx = read_buff[RAW_IJKL_TO_N(nx,ny,nz,i,j,k,0)],
          fy = read_buff[RAW_IJKL_TO_N(nx,ny,nz,i,j,k,1)],
          fz = read_buff[RAW_IJKL_TO_N(nx,ny,nz,i,j,k,2)];

        lat->set_force_ijk(i, j, k, fx, fy, fz);
      }
    }
  }
  delete [] read_buff;
  fclose(ifile);
  return true;
}
//---------------------------------------------------------------------------
bool LB_Raw_IO::write_frc(const char *base_fname, Lattice::Isothermal3D *lat)
{
  FILE *ofile;
  if( !open_file(base_fname, ".force.out.raw", "wb", &ofile) ) return false;
  
  unsigned int nx = lat->nx(), ny = lat->ny(),
    nz = lat->nz(), ncount = 3*nx*ny*nz;
  double *write_buff = new double[ncount];
    
  #pragma omp parallel for
  for(unsigned int k = 0; k < nz; ++k) {
    for(unsigned int j = 0; j < ny; ++j) {
      for(unsigned int i = 0; i < nx; ++i)
      {
        double fx,fy,fz;
        lat->get_force_ijk(i,j,k,&fx,&fy,&fz);
        write_buff[RAW_IJKL_TO_N(nx, ny, nz, i, j, k, 0)] = fx; 
        write_buff[RAW_IJKL_TO_N(nx, ny, nz, i, j, k, 1)] = fy; 
        write_buff[RAW_IJKL_TO_N(nx, ny, nz, i, j, k, 2)] = fz; 
      }
    }
  }
  fwrite(write_buff, sizeof(double), ncount, ofile);

  delete [] write_buff;;
  fclose(ofile);
  return true;
}
//===========================================================================