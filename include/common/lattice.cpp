//===========================================================================
// Author: Keijo Mattila, JYU, February 2015
// Description: Implementation of Lattice data structures
// Details: Lattices store geometries + hydrodynamic variables,
//          but not distributions values
//===========================================================================
#include <limits>
#include <stdio.h>
#include "lattice.h"
#include "lb_const.h"
//===========================================================================
using namespace Lattice;
//===========================================================================
// Class SparseGeoml3D
//---------------------------------------------------------------------------
void SparseGeom3D::alloc_geom(UINT nx, UINT ny, UINT nz)
{
  unsigned long long int ull_nx = nx, ull_ny = ny, ull_nz = nz,
    ull_ushort_max = std::numeric_limits<unsigned short>::max(),
    ull_uint_max = std::numeric_limits<UINT>::max(),
    ull_ncount = ull_nx*ull_ny*ull_nz;
    
  if( ull_ncount == 0 || ull_ncount >= ull_uint_max ) {
    printf("Require: 0 < lattice node count < UINT max!\n");
    printf("Warning: Lattice size set to nx = 1, ny = 1, nz = 1!\n");
    nx = 1; ny = 1; nz = 1;
  }
  if( ull_nx > ull_ushort_max || ull_ny > ull_ushort_max ||
      ull_nz > ull_ushort_max ) {
    printf("Require: nx, ny, nz <= unsigned short max)!\n");
    printf("Warning: Lattice size set to nx = 1, ny = 1, nz = 1!\n");
    nx = 1; ny = 1; nz = 1;
  }
  a_nx = nx; a_ny = ny; a_nz = nz;
  UINT ncount = a_nx*a_ny*a_nz;

  a_geom = new unsigned char[ncount];
  a_fnode_enum = new UINT[ncount];
  
  #pragma omp parallel for
  for(UINT n = 0; n < ncount; ++n) {
    a_geom[n] = LB_Const::SOLID_NODE;
    a_fnode_enum[n] = 0;
  }
}
//---------------------------------------------------------------------------
void SparseGeom3D::alloc_lat_coords(UINT fluid_ncount)
{
  a_fluid_ncount = fluid_ncount;
  UINT data_ncount = fluid_ncount + 1;

  a_li = new unsigned short[data_ncount];
  a_lj = new unsigned short[data_ncount];
  a_lk = new unsigned short[data_ncount];

  #pragma omp parallel for
  for(UINT k = 0; k < a_nz; ++k) {
    for(UINT j = 0; j < a_ny; ++j) {
      for(UINT i = 0; i < a_nx; ++i)
      {
        UINT n = IJK_TO_N(i,j,k);
        if( a_geom[n] == LB_Const::SOLID_NODE ) continue;

        unsigned int dn = a_fnode_enum[n];
        a_li[dn] = i; a_lj[dn] = j; a_lk[dn] = k;
      }
    }
  }
  a_li[0] = 0; a_lj[0] = 0; a_lk[0] = 0;
}
//===========================================================================
void SparseGeom3D::dealloc_geom()
{
  if( a_geom != 0 ) {
    delete [] a_geom;
    a_geom = 0;
  }
  if( a_fnode_enum != 0 ) {
    delete [] a_fnode_enum;
    a_fnode_enum = 0;
  }
  if( a_li != 0 ) {
    delete [] a_li;
    a_li = 0;
  }
  if( a_lj != 0 ) {
    delete [] a_lj;
    a_lj = 0;
  }
  if( a_lk != 0 ) {
    delete [] a_lk;
    a_lk = 0;
  }
  a_nx = 0; a_ny = 0; a_nz = 0;
}
//---------------------------------------------------------------------------
void SparseGeom3D::dealloc_lat_coords()
{
  if( a_li != 0 ) {
    delete [] a_li;
    a_li = 0;
  }
  if( a_lj != 0 ) {
    delete [] a_lj;
    a_lj = 0;
  }
  if( a_lk != 0 ) {
    delete [] a_lk;
    a_lk = 0;
  }
  a_fluid_ncount = 0;
}
//---------------------------------------------------------------------------
SparseGeom3D::SparseGeom3D(UINT nx, UINT ny, UINT nz)
{
  alloc_geom(nx, ny, nz);
  alloc_lat_coords(0);
}
//---------------------------------------------------------------------------
SparseGeom3D::~SparseGeom3D()
{
  dealloc_lat_coords();
  dealloc_geom();
}
//---------------------------------------------------------------------------
void SparseGeom3D::set_geom(unsigned char *geom)
{
  dealloc_fluid_data();
  dealloc_lat_coords();
  UINT fluid_ncount = 0;
  
  for(UINT k = 0; k < a_nz; ++k) {
    for(UINT j = 0; j < a_ny; ++j) {
      for(UINT i = 0; i < a_nx; ++i)
      {
        UINT n = IJK_TO_N(i,j,k);
        a_geom[n] = geom[n];
        
        if( a_geom[n] == LB_Const::SOLID_NODE ) {
          a_fnode_enum[n] = 0;
        } else {
          a_fnode_enum[n] = fluid_ncount + 1;
          fluid_ncount++;
        }
      }
    }
  }
  alloc_lat_coords(fluid_ncount);
  alloc_fluid_data();
}
//---------------------------------------------------------------------------
// Class Isothermal3D
//---------------------------------------------------------------------------
Isothermal3D::Isothermal3D(UINT nx, UINT ny, UINT nz): SparseGeom3D(nx,ny,nz)
{
  alloc_fluid_data();
}
//---------------------------------------------------------------------------
Isothermal3D::~Isothermal3D()
{
  dealloc_fluid_data();
}
//---------------------------------------------------------------------------
void Isothermal3D::dealloc_fluid_data()
{
  if( a_den != 0 ) {
    delete [] a_den;
    a_den = 0;
  }
  if( a_ux != 0 ) {
    delete [] a_ux;
    a_ux = 0;
  }
  if( a_uy != 0 ) {
    delete [] a_uy;
    a_uy = 0;
  }
  if( a_uz != 0 ) {
    delete [] a_uz;
    a_uz = 0;
  }
  if( a_fx != 0 ) {
    delete [] a_fx;
    a_fx = 0;
  }
  if( a_fy != 0 ) {
    delete [] a_fy;
    a_fy = 0;
  }
  if( a_fz != 0 ) {
    delete [] a_fz;
    a_fz = 0;
  }
}
//---------------------------------------------------------------------------
void Isothermal3D::alloc_fluid_data()
{
  UINT data_ncount = a_fluid_ncount + 1;
  a_den = new double[data_ncount];
  a_ux = new double[data_ncount];
  a_uy = new double[data_ncount];
  a_uz = new double[data_ncount];
  a_fx = new double[data_ncount];
  a_fy = new double[data_ncount];
  a_fz = new double[data_ncount];

  #pragma omp parallel for
  for(UINT k = 0; k < a_nz; ++k) {
    for(UINT j = 0; j < a_ny; ++j) {
      for(UINT i = 0; i < a_nx; ++i)
      {
        UINT n = IJK_TO_N(i,j,k);
        if( a_geom[n] == LB_Const::SOLID_NODE ) continue;

        unsigned int dn = a_fnode_enum[n];
        a_den[dn] = 0.0;
        a_ux[dn] = 0.0; a_uy[dn] = 0.0; a_uz[dn] = 0.0;
        a_fx[dn] = 0.0; a_fy[dn] = 0.0; a_fz[dn] = 0.0;
      }
    }
  }
  a_den[0] = 0.0;
  a_ux[0] = 0.0; a_uy[0] = 0.0; a_uz[0] = 0.0;
  a_fx[0] = 0.0; a_fy[0] = 0.0; a_fz[0] = 0.0;
}
//===========================================================================