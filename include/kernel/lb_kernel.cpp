//===========================================================================
// Author: Keijo Mattila, JYU, February 2015
// Description: Implementation of LBM kernels
// Details: Computation in dimensionless variables,
//          input/output with dimensional variables;
//          AA-pattern algorithm, TRT-collision operator
//===========================================================================
#include <math.h>
#include <limits>
#include <stdio.h>
#include "D3Q19.h"
#include "lb_kernel.h"
//===========================================================================
using namespace D3Q19;
using namespace LB_Kernel;
//===========================================================================
typedef void (Isothermal3D::*FI_IND_FPTR)(ULLINT *, ULLINT);
//===========================================================================
void Isothermal3D::get_fi_ind_arr_odd_t(ULLINT *fi_ind, ULLINT fnode_enum)
{
  ULLINT off = 19ull*fnode_enum;
  fi_ind[BS] = a_mem_addr[off];
  fi_ind[BW] = a_mem_addr[off + 1ull];
  fi_ind[B]  = a_mem_addr[off + 2ull];
  fi_ind[BE] = a_mem_addr[off + 3ull];
  fi_ind[BN] = a_mem_addr[off + 4ull];
  fi_ind[SW] = a_mem_addr[off + 5ull];
  fi_ind[S]  = a_mem_addr[off + 6ull];
  fi_ind[SE] = a_mem_addr[off + 7ull];
  fi_ind[W]  = a_mem_addr[off + 8ull];
  fi_ind[C]  = a_mem_addr[off + 9ull];
  fi_ind[E]  = a_mem_addr[off + 10ull];
  fi_ind[NW] = a_mem_addr[off + 11ull];
  fi_ind[N]  = a_mem_addr[off + 12ull];
  fi_ind[NE] = a_mem_addr[off + 13ull];
  fi_ind[TS] = a_mem_addr[off + 14ull];
  fi_ind[TW] = a_mem_addr[off + 15ull];
  fi_ind[T]  = a_mem_addr[off + 16ull];
  fi_ind[TE] = a_mem_addr[off + 17ull];
  fi_ind[TN] = a_mem_addr[off + 18ull];
}
//---------------------------------------------------------------------------
void Isothermal3D::get_fi_ind_arr_even_t(ULLINT *fi_ind, ULLINT fnode_enum)
{
  fi_ind[BS] = F_TN_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[BW] = F_TE_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[B]  = F_T_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[BE] = F_TW_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[BN] = F_TS_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[SW] = F_NE_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[S]  = F_N_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[SE] = F_NW_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[W]  = F_E_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[C]  = F_C_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[E]  = F_W_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[NW] = F_SE_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[N]  = F_S_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[NE] = F_SW_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[TS] = F_BN_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[TW] = F_BE_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[T]  = F_B_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[TE] = F_BW_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[TN] = F_BS_IND(fnode_enum,a_ull_fluid_ncount);
}
//---------------------------------------------------------------------------
void Isothermal3D::get_fi_ind_arr(ULLINT *fi_ind, ULLINT fnode_enum)
{
  fi_ind[BS] = F_BS_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[BW] = F_BW_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[B]  = F_B_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[BE] = F_BE_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[BN] = F_BN_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[SW] = F_SW_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[S]  = F_S_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[SE] = F_SE_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[W]  = F_W_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[C]  = F_C_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[E]  = F_E_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[NW] = F_NW_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[N]  = F_N_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[NE] = F_NE_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[TS] = F_TS_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[TW] = F_TW_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[T]  = F_T_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[TE] = F_TE_IND(fnode_enum,a_ull_fluid_ncount);
  fi_ind[TN] = F_TN_IND(fnode_enum,a_ull_fluid_ncount);
}
//---------------------------------------------------------------------------
ULLINT Isothermal3D::get_fi_ind(ULLINT fnode_enum, unsigned char l)
{
  switch( l ) {
    case BS: return F_BS_IND(fnode_enum,a_ull_fluid_ncount);
    case BW: return F_BW_IND(fnode_enum,a_ull_fluid_ncount);
    case B:  return F_B_IND(fnode_enum,a_ull_fluid_ncount);
    case BE: return F_BE_IND(fnode_enum,a_ull_fluid_ncount);
    case BN: return F_BN_IND(fnode_enum,a_ull_fluid_ncount);
    case SW: return F_SW_IND(fnode_enum,a_ull_fluid_ncount);
    case S:  return F_S_IND(fnode_enum,a_ull_fluid_ncount);
    case SE: return F_SE_IND(fnode_enum,a_ull_fluid_ncount);
    case W:  return F_W_IND(fnode_enum,a_ull_fluid_ncount);
    case C:  return F_C_IND(fnode_enum,a_ull_fluid_ncount);
    case E:  return F_E_IND(fnode_enum,a_ull_fluid_ncount);
    case NW: return F_NW_IND(fnode_enum,a_ull_fluid_ncount);
    case N:  return F_N_IND(fnode_enum,a_ull_fluid_ncount);
    case NE: return F_NE_IND(fnode_enum,a_ull_fluid_ncount);
    case TS: return F_TS_IND(fnode_enum,a_ull_fluid_ncount);
    case TW: return F_TW_IND(fnode_enum,a_ull_fluid_ncount);
    case T:  return F_T_IND(fnode_enum,a_ull_fluid_ncount);
    case TE: return F_TE_IND(fnode_enum,a_ull_fluid_ncount);
    case TN: return F_TN_IND(fnode_enum,a_ull_fluid_ncount);
    default: return F_C_IND(fnode_enum,a_ull_fluid_ncount);
  } 
}
//---------------------------------------------------------------------------
void Isothermal3D::alloc_memory(unsigned int fluid_ncount)
{
  a_ull_fluid_ncount = fluid_ncount;
  ULLINT mem_count_d3q19 = a_ull_fluid_ncount*(ULLINT)(Q);

  a_mem_addr = new ULLINT[mem_count_d3q19];
  a_fi = new double[mem_count_d3q19];

  #pragma omp parallel for
  for(ULLINT m = 0ull; m < mem_count_d3q19; ++m) {
    a_mem_addr[m] = 0;
    a_fi[m] = 0.0;
  }
}
//---------------------------------------------------------------------------
void Isothermal3D::dealloc_memory()
{
  if( a_mem_addr != 0 ) {
    delete [] a_mem_addr;
    a_mem_addr = 0;
  }
  if( a_fi != 0 ) {
    delete [] a_fi;
    a_fi = 0;
  }
  if( a_coll_oper != 0 ) {
    delete a_coll_oper;
    a_coll_oper = 0;
  }
  a_ull_fluid_ncount = 0;
}
//---------------------------------------------------------------------------
void Isothermal3D::config_evolve(Lattice::Isothermal3D *lat)
{
  using namespace LB_Const;
  unsigned int fluid_ncount = lat->fluid_ncount();
  int nx = lat->nx(), ny = lat->ny(), nz = lat->nz();

  alloc_memory(fluid_ncount);
  
  #pragma omp parallel for
  for(unsigned int fnode_enum = 0; fnode_enum < fluid_ncount; ++fnode_enum)
  {
    unsigned int i,j,k;
    lat->get_fnode_lat_coord(fnode_enum, &i, &j, &k);

    for(unsigned char l = 0; l < Q; ++l)
    {
      // Periodic bc in all directions by default
      int ci = CI[l][X], cj = CI[l][Y], ck = CI[l][Z],
        ni = (int)i - ci, nj = (int)j - cj, nk = (int)k - ck,
        per_ni = (ni+nx)%nx, per_nj = (nj+ny)%ny, per_nk = (nk+nz)%nz;
            
      unsigned int fnode_enum_nn;
      ULLINT mem_addr_ind = 19ull*(ULLINT)fnode_enum + (ULLINT)l;
                     
      // Halfway-bounceback
      if( lat->get_fnode_enum_ijk(per_ni, per_nj, per_nk, &fnode_enum_nn) ) {
        a_mem_addr[mem_addr_ind] = get_fi_ind(fnode_enum_nn,l);
      } else {
        a_mem_addr[mem_addr_ind] = get_fi_ind(fnode_enum,RDIR[l]);
      }
    }
  }
  a_odd_t = true;
}
//---------------------------------------------------------------------------
void Isothermal3D::comp_der(Lattice::Isothermal3D *lat,
  unsigned int back_nn, unsigned int forw_nn, double ux, double uy,
  double uz, double *der_ux, double *der_uy, double *der_uz)
{
  if( lat->get_geom_n(forw_nn) == LB_Const::SOLID_NODE ) {
    if( lat->get_geom_n(back_nn) == LB_Const::SOLID_NODE ) {
      // der_g = [g(+0.5) - g(-0.5)]/(2*0.5)
      (*der_ux) = 0.0;
      (*der_uy) = 0.0;
      (*der_uz) = 0.0;
    } else {
      // der_g = [4*g(0.5) - 3*g(0) - g(-1)]/3
      double ux_back, uy_back, uz_back;
      lat->get_vel_n(back_nn,&ux_back,&uy_back,&uz_back);
      ux_back *= a_inv_cr; uy_back *= a_inv_cr; uz_back *= a_inv_cr;

      (*der_ux) = (1.0/3.0)*(4.0*0.0 - 3.0*ux - ux_back);
      (*der_uy) = (1.0/3.0)*(4.0*0.0 - 3.0*uy - uy_back);
      (*der_uz) = (1.0/3.0)*(4.0*0.0 - 3.0*uz - uz_back);
    }
  } else if( lat->get_geom_n(back_nn) == LB_Const::SOLID_NODE ) {
    // der_g = [g(+1) + 3*g(0) - 4*g(-0.5)]/3
    double ux_forw, uy_forw, uz_forw;
    lat->get_vel_n(forw_nn,&ux_forw,&uy_forw,&uz_forw);
    ux_forw *= a_inv_cr; uy_forw *= a_inv_cr; uz_forw *= a_inv_cr;

    (*der_ux) = (1.0/3.0)*(ux_forw + 3.0*ux - 4.0*0.0);
    (*der_uy) = (1.0/3.0)*(uy_forw + 3.0*uy - 4.0*0.0);
    (*der_uz) = (1.0/3.0)*(uz_forw + 3.0*uz - 4.0*0.0);
  } else {
    // der_g = [g(+1) - g(-1)]/(2*1)
    double ux_back, uy_back, uz_back, ux_forw, uy_forw, uz_forw;
    lat->get_vel_n(back_nn,&ux_back,&uy_back,&uz_back);
    lat->get_vel_n(forw_nn,&ux_forw,&uy_forw,&uz_forw);
    ux_back *= a_inv_cr; uy_back *= a_inv_cr; uz_back *= a_inv_cr;
    ux_forw *= a_inv_cr; uy_forw *= a_inv_cr; uz_forw *= a_inv_cr;

    (*der_ux) = 0.5*(ux_forw - ux_back);
    (*der_uy) = 0.5*(uy_forw - uy_back);
    (*der_uz) = 0.5*(uz_forw - uz_back);
  }
}
//---------------------------------------------------------------------------
void Isothermal3D::initialize_fi(Lattice::Isothermal3D *lat)
{
  using namespace LB_Const;

  ULLINT fi_ind[Q];
  double fi[Q], feq[Q], fext[Q], KI2[Q][3][3],
    pab_cff = -2.0*a_coll_oper->get_kvisc_relax_t()*CT2,
    gx = a_dt*a_inv_cr*a_gx, gy = a_dt*a_inv_cr*a_gy,
    gz = a_dt*a_inv_cr*a_gz;
  
  unsigned int fluid_ncount = lat->fluid_ncount();
  int nx = lat->nx(), ny = lat->ny(), nz = lat->nz();
  get_kin_proj2(KI2);
  
  #pragma omp parallel for private(fi_ind, fi, feq, fext)
  for(unsigned int fnode_enum = 0; fnode_enum < fluid_ncount; ++fnode_enum)
  {
    unsigned int i,j,k;
    get_fi_ind_arr(fi_ind, fnode_enum);
    lat->get_fnode_lat_coord(fnode_enum, &i, &j, &k);
        
    double den, ux, uy, uz, fx, fy, fz;
    lat->get_fnode_den(fnode_enum, &den);
    lat->get_fnode_vel(fnode_enum, &ux, &uy, &uz);
    lat->get_fnode_force(fnode_enum,&fx,&fy,&fz);

    den *= a_inv_ref_den; ux *= a_inv_cr; uy *= a_inv_cr; uz *= a_inv_cr;
    fx *= a_inv_bf_unit; fy *= a_inv_bf_unit; fz *= a_inv_bf_unit;
    fx += den*gx; fy += den*gy; fz += den*gz;

    a_frc_func(ux,uy,uz,fx,fy,fz,fext);
    a_eq_func(den,ux,uy,uz,feq);
        
    int per_i_e = ((int)i + 1 + nx)%nx, per_i_w = ((int)i - 1 + nx)%nx,
        per_j_n = ((int)j + 1 + ny)%ny, per_j_s = ((int)j - 1 + ny)%ny,
        per_k_t = ((int)k + 1 + nz)%nz, per_k_b = ((int)k - 1 + nz)%nz;

    unsigned int nn_e = lat->IJK_TO_N(per_i_e,j,k),
      nn_w = lat->IJK_TO_N(per_i_w,j,k),
      nn_n = lat->IJK_TO_N(i,per_j_n,k),
      nn_s = lat->IJK_TO_N(i,per_j_s,k),
      nn_t = lat->IJK_TO_N(i,j,per_k_t),
      nn_b = lat->IJK_TO_N(i,j,per_k_b);

    double dx_ux, dy_ux, dz_ux, dx_uy, dy_uy, dz_uy, dx_uz, dy_uz, dz_uz;
    comp_der(lat, nn_w, nn_e, ux, uy, uz, &dx_ux, &dx_uy, &dx_uz);
    comp_der(lat, nn_s, nn_n, ux, uy, uz, &dy_ux, &dy_uy, &dy_uz);
    comp_der(lat, nn_b, nn_t, ux, uy, uz, &dz_ux, &dz_uy, &dz_uz);

    double P_xx = pab_cff*den*dx_ux,
      P_xy = pab_cff*den*0.5*(dx_uy + dy_ux),
      P_xz = pab_cff*den*0.5*(dx_uz + dz_ux),
      P_yy = pab_cff*den*dy_uy,
      P_yz = pab_cff*den*0.5*(dy_uz + dz_uy),
      P_zz = pab_cff*den*dz_uz;

    for(unsigned char l = 0; l < Q; ++l)
    {
      double fneq = KI2[l][X][X]*P_xx + KI2[l][Y][Y]*P_yy + KI2[l][Z][Z]*P_zz
        + 2.0*(KI2[l][X][Y]*P_xy + KI2[l][X][Z]*P_xz + KI2[l][Y][Z]*P_yz);
      
      fi[l] = feq[l] + fneq;
    }
    a_coll_oper->relax(fi, feq, fext);
    for(unsigned char l = 0; l < Q; ++l) {a_fi[fi_ind[l]] = fi[l];}
	}		
}
//---------------------------------------------------------------------------
Isothermal3D::Isothermal3D(Lattice::Isothermal3D *lat, double ref_den,
  double dr, double dt, double kvisc, double gx, double gy, double gz,
  unsigned char flow_type, unsigned char coll_oper, bool external_forcing)
{
  config_evolve(lat);

  a_ref_den = ref_den; a_dr = dr; a_dt = dt;
  a_inv_ref_den = 1.0/a_ref_den; a_inv_dr = 1.0/a_dr; a_inv_dt = 1.0/a_dt;

  a_cr = a_dr*a_inv_dt; a_inv_cr = 1.0/a_cr;
  a_bf_unit = a_ref_den*a_inv_dt*a_cr;
  a_inv_bf_unit = 1.0/a_bf_unit;

  if( flow_type == LB_Const::STOKES_FLOW ) {a_eq_func = &eq1;}
  else {a_eq_func = &eq2;}

  if( external_forcing ) {a_frc_func = &D3Q19::frc2;}
  else {a_frc_func = &D3Q19::frc1;}
    
  switch( coll_oper ) {
    case LB_Const::BGK: {
      a_coll_oper = new LB_Collision::BGK_D3Q19();
      break;}
    case LB_Const::TRT: {
      a_coll_oper = new LB_Collision::TRT_D3Q19();
      break;}
//    case LB_Const::MRT: {
//      a_coll_oper = new LB_Collision::MRT_D3Q19();
//      break;}
//    case LB_Const::REG: {
//      a_coll_oper = new LB_Collision::REG_D3Q19();
//      break;}
    default: {
      printf("Warning: using default collision operator (TRT)\n");
      a_coll_oper = new LB_Collision::TRT_D3Q19();}
  } 
  set_gravity(gx, gy, gz);
  set_kvisc(kvisc);
  a_cs = a_cr*CT;

  initialize_fi(lat);
}
//---------------------------------------------------------------------------
Isothermal3D::~Isothermal3D()
{
  dealloc_memory();
}
//---------------------------------------------------------------------------
void Isothermal3D::set_kvisc(double kvisc)
{
  a_kvisc = kvisc;
  double dimless_kvisc = a_inv_dr*a_inv_cr*kvisc;

  a_coll_oper->set_kvisc(dimless_kvisc);  
}
//---------------------------------------------------------------------------
void Isothermal3D::set_gravity(double gx, double gy, double gz)
{
  a_gx = gx; a_gy = gy; a_gz = gz;
}
//---------------------------------------------------------------------------
void Isothermal3D::evolve(Lattice::Isothermal3D *lat)
{
  ULLINT fi_ind[Q];
  unsigned int fluid_ncount = lat->fluid_ncount();
  double fi[Q], feq[Q], fext[Q], gx = a_dt*a_inv_cr*a_gx,
    gy = a_dt*a_inv_cr*a_gy, gz = a_dt*a_inv_cr*a_gz;

  FI_IND_FPTR fi_ind_func = &Isothermal3D::get_fi_ind_arr_even_t;
  if( a_odd_t ) fi_ind_func = &Isothermal3D::get_fi_ind_arr_odd_t;

  #pragma omp parallel for private(fi_ind, fi, feq, fext)
  for(unsigned int fnode_enum = 0; fnode_enum < fluid_ncount; ++fnode_enum)
  {
    (this->*fi_ind_func)(fi_ind, fnode_enum);

    fi[BS] = a_fi[fi_ind[BS]];
    fi[BW] = a_fi[fi_ind[BW]];
    fi[B]  = a_fi[fi_ind[B]];
    fi[BE] = a_fi[fi_ind[BE]];
    fi[BN] = a_fi[fi_ind[BN]];
    fi[SW] = a_fi[fi_ind[SW]];
    fi[S]  = a_fi[fi_ind[S]];
    fi[SE] = a_fi[fi_ind[SE]];
    fi[W]  = a_fi[fi_ind[W]];
    fi[C]  = a_fi[fi_ind[C]];
    fi[E]  = a_fi[fi_ind[E]];
    fi[NW] = a_fi[fi_ind[NW]];
    fi[N]  = a_fi[fi_ind[N]];
    fi[NE] = a_fi[fi_ind[NE]];
    fi[TS] = a_fi[fi_ind[TS]];
    fi[TW] = a_fi[fi_ind[TW]];
    fi[T]  = a_fi[fi_ind[T]];
    fi[TE] = a_fi[fi_ind[TE]];
    fi[TN] = a_fi[fi_ind[TN]];

    double den, ux, uy, uz, fx, fy, fz,
      inv_den = den_vel_moms(fi, &den, &ux, &uy, &uz);

    lat->get_fnode_force(fnode_enum,&fx,&fy,&fz);
    fx *= a_inv_bf_unit; fy *= a_inv_bf_unit; fz *= a_inv_bf_unit;
    fx += den*gx; fy += den*gy; fz += den*gz;

    ux += 0.5*inv_den*fx; uy += 0.5*inv_den*fy; uz += 0.5*inv_den*fz;
    a_frc_func(ux,uy,uz,fx,fy,fz,fext);
    a_eq_func(den,ux,uy,uz,feq);

    den *= a_ref_den; ux *= a_cr; uy *= a_cr; uz *= a_cr;
    lat->set_fnode_den(fnode_enum,den);
    lat->set_fnode_vel(fnode_enum,ux,uy,uz);

    a_coll_oper->relax(fi, feq, fext);
    
    // Store new distributions into opposite directions (AA-pattern)
    a_fi[fi_ind[BS]] = fi[TN];
    a_fi[fi_ind[BW]] = fi[TE];
    a_fi[fi_ind[B]]  = fi[T];
    a_fi[fi_ind[BE]] = fi[TW];
    a_fi[fi_ind[BN]] = fi[TS];
    a_fi[fi_ind[SW]] = fi[NE];
    a_fi[fi_ind[S]]  = fi[N];
    a_fi[fi_ind[SE]] = fi[NW];
    a_fi[fi_ind[W]]  = fi[E];
    a_fi[fi_ind[C]]  = fi[C];
    a_fi[fi_ind[E]]  = fi[W];
    a_fi[fi_ind[NW]] = fi[SE];
    a_fi[fi_ind[N]]  = fi[S];
    a_fi[fi_ind[NE]] = fi[SW];
    a_fi[fi_ind[TS]] = fi[BN];
    a_fi[fi_ind[TW]] = fi[BE];
    a_fi[fi_ind[T]]  = fi[B];
    a_fi[fi_ind[TE]] = fi[BW];
    a_fi[fi_ind[TN]] = fi[BS];
  }
  a_odd_t = !a_odd_t;
}
//===========================================================================