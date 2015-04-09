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
#include "kernel.h"
//===========================================================================
using namespace D3Q19;
//===========================================================================
// bit masks for integer composed of at least 32 bits
const UINT BIT1_MASK  = 1;          // 2^0
const UINT BIT2_MASK  = 2;          // 2^1
const UINT BIT3_MASK  = 4;          // 2^2
const UINT BIT4_MASK  = 8;          // 2^3
const UINT BIT5_MASK  = 16;         // 2^4
const UINT SW_MASK = 32;         // 2^5
const UINT BW_MASK = 64;         // 2^6
const UINT W_MASK  = 128;        // 2^7
const UINT TW_MASK = 256;        // 2^8
const UINT NW_MASK = 512;        // 2^9
const UINT BS_MASK = 1024;       // 2^10
const UINT S_MASK  = 2048;       // 2^11
const UINT TS_MASK = 4096;       // 2^12
const UINT B_MASK  = 8192;       // 2^13
const UINT C_MASK  = 16384;      // 2^14
const UINT T_MASK  = 32768;      // 2^15
const UINT BN_MASK = 65536;      // 2^16
const UINT N_MASK  = 131072;     // 2^17
const UINT TN_MASK = 262144;     // 2^18
const UINT SE_MASK = 524288;     // 2^19
const UINT BE_MASK = 1048576;    // 2^20
const UINT E_MASK  = 2097152;    // 2^21
const UINT TE_MASK = 4194304;    // 2^22
const UINT NE_MASK = 8388608;    // 2^23
const UINT BIT25_MASK = 16777216;   // 2^24
const UINT BIT26_MASK = 33554432;   // 2^25
const UINT BIT27_MASK = 67108864;   // 2^26
const UINT BIT28_MASK = 134217728;  // 2^27
const UINT BIT29_MASK = 268435456;  // 2^28
const UINT BIT30_MASK = 536870912;  // 2^29
const UINT BIT31_MASK = 1073741824; // 2^30
const UINT BIT32_MASK = 2147483648; // 2^31

// in FULL_MASK every bit has value 1
const UINT FULL_MASK = 0xFFFFFFFF;
// in QCOUNT_MASK first five bits have value 1
const UINT QCOUNT_MASK = 0x0000001F;
// in OPTIONAL_BYTE_MASK last eight bits have value 1
const UINT OPTIONAL_BYTE_MASK = 0xFF000000;

UINT CI_MASKS[Q];
//===========================================================================
typedef void (IsothermalKernel::*FI_IND_FPTR)(ULLINT *, ULLINT);
//===========================================================================
void init_ci_masks()
{
  CI_MASKS[BS] = BS_MASK;
  CI_MASKS[BW] = BW_MASK;
  CI_MASKS[B]  = B_MASK;
  CI_MASKS[BE] = BE_MASK;
  CI_MASKS[BN] = BN_MASK;
  CI_MASKS[SW] = SW_MASK;
  CI_MASKS[S]  = S_MASK;
  CI_MASKS[SE] = SE_MASK;
  CI_MASKS[W]  = W_MASK;
  CI_MASKS[C]  = C_MASK;
  CI_MASKS[E]  = E_MASK;
  CI_MASKS[NW] = NW_MASK;
  CI_MASKS[N]  = N_MASK;
  CI_MASKS[NE] = NE_MASK;
  CI_MASKS[TS] = TS_MASK;
  CI_MASKS[TW] = TW_MASK;
  CI_MASKS[T]  = T_MASK;
  CI_MASKS[TE] = TE_MASK;
  CI_MASKS[TN] = TN_MASK;
}
//---------------------------------------------------------------------------
bool IsothermalKernel::is_solid_nghbr(UINT fnode_enum, unsigned char l)
{
  return (_nghbr_info[fnode_enum] & CI_MASKS[l]) != 0;
}
//---------------------------------------------------------------------------
UINT IsothermalKernel::get_solid_nghbr_count(UINT fnode_enum)
{
  return (_nghbr_info[fnode_enum] & QCOUNT_MASK);
}
//---------------------------------------------------------------------------
void IsothermalKernel::get_fi_ind_arr_odd_t(ULLINT *fi_ind, ULLINT fnode_enum)
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
void IsothermalKernel::get_fi_ind_arr_even_t(ULLINT *fi_ind, ULLINT fnode_enum)
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
void IsothermalKernel::get_fi_ind_arr(ULLINT *fi_ind, ULLINT fnode_enum)
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
ULLINT IsothermalKernel::get_fi_ind(ULLINT fnode_enum, unsigned char l)
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
void IsothermalKernel::alloc_memory(UINT fluid_ncount)
{
  a_ull_fluid_ncount = fluid_ncount;
  ULLINT mem_count_d3q19 = a_ull_fluid_ncount*(ULLINT)(Q);

  a_mem_addr = new ULLINT[mem_count_d3q19];
  a_fi = new double[mem_count_d3q19];
  _nghbr_info = new UINT[fluid_ncount];

  #pragma omp parallel for
  for(ULLINT m = 0ull; m < mem_count_d3q19; ++m) {
    a_mem_addr[m] = 0;
    a_fi[m] = 0.0;
  }
}
//---------------------------------------------------------------------------
void IsothermalKernel::dealloc_memory()
{
  if( a_mem_addr != 0 ) {
    delete [] a_mem_addr;
    a_mem_addr = 0;
  }
  if( a_fi != 0 ) {
    delete [] a_fi;
    a_fi = 0;
  }
  if( _nghbr_info != 0 ) {
    delete [] _nghbr_info;
    _nghbr_info = 0;
  }
  if( a_coll_oper != 0 ) {
    delete a_coll_oper;
    a_coll_oper = 0;
  }
  a_ull_fluid_ncount = 0;
}
//---------------------------------------------------------------------------
void IsothermalKernel::config_evolve(Geometry *geom, NodeSet *fnodes)
{
  Lattice *lat = geom->get_lattice();
  GeomData *phase = geom->get_phase();
  
  UINT size[3], fluid_ncount = fnodes->get_node_count();
  lat->get_size(size);

  _nx = size[0]; _ny = size[1]; _nz = size[2];
  alloc_memory(fluid_ncount);
  
  #pragma omp parallel for
  for(UINT fnode_enum = 0; fnode_enum < fluid_ncount; ++fnode_enum)
  {
    UINT ijk[3];
    fnodes->get_ijk(fnode_enum, ijk);

    UINT solid_nghbr_count = 0;
    _nghbr_info[fnode_enum] = 0;
    for(unsigned char l = 0; l < Q; ++l)
    {
      // Periodic bc in all directions by default
      int ci = CI[l][X], cj = CI[l][Y], ck = CI[l][Z],
        ni = (int)ijk[0] - ci, nj = (int)ijk[1] - cj, nk = (int)ijk[2] - ck,
        per_ni = (ni + (int)_nx)%(int)_nx,
        per_nj = (nj + (int)_ny)%(int)_ny,
        per_nk = (nk + (int)_nz)%(int)_nz;
            
      UINT ijk_n[3] = {(UINT)per_ni, (UINT)per_nj, (UINT)per_nk};
      ULLINT mem_addr_ind = 19ull*(ULLINT)fnode_enum + (ULLINT)l;
               
      // Halfway-bounceback (FLUID_NODE->stream, SOLID_NODE->hbb)
      if( phase->get_val_ijk(ijk_n) == FLUID_NODE ) {
        UINT fnode_enum_nn = fnodes->get_n(ijk_n);
        a_mem_addr[mem_addr_ind] = get_fi_ind(fnode_enum_nn,l);
      } else {
        a_mem_addr[mem_addr_ind] = get_fi_ind(fnode_enum,RDIR[l]);
        _nghbr_info[fnode_enum] |= CI_MASKS[RDIR[l]];
        solid_nghbr_count++;
      }
    }
    _nghbr_info[fnode_enum] |= solid_nghbr_count;
  }
  a_odd_t = true;
}
//---------------------------------------------------------------------------
void IsothermalKernel::comp_der(IsothermalNodeData *fdata, UINT fnode_enum, 
  unsigned char forw_l, UINT ijk_forw[3], UINT ijk_back[3],
  double ux, double uy, double uz, double *der_ux,
  double *der_uy, double *der_uz)
{
  unsigned char back_l = RDIR[forw_l];
  FieldData *vx = fdata->velx(), *vy = fdata->vely(), *vz = fdata->velz();

  if( is_solid_nghbr(fnode_enum, forw_l) ) {
    if( is_solid_nghbr(fnode_enum, back_l) ) {
      // der_g = [g(+0.5) - g(-0.5)]/(2*0.5)
      (*der_ux) = 0.0;
      (*der_uy) = 0.0;
      (*der_uz) = 0.0;
    } else {
      // der_g = [4*g(0.5) - 3*g(0) - g(-1)]/3
      double ux_back = vx->get_val_ijk(ijk_back),
        uy_back = vy->get_val_ijk(ijk_back),
        uz_back = vz->get_val_ijk(ijk_back);

      ux_back *= a_inv_cr;
      uy_back *= a_inv_cr;
      uz_back *= a_inv_cr;

      (*der_ux) = (1.0/3.0)*(4.0*0.0 - 3.0*ux - ux_back);
      (*der_uy) = (1.0/3.0)*(4.0*0.0 - 3.0*uy - uy_back);
      (*der_uz) = (1.0/3.0)*(4.0*0.0 - 3.0*uz - uz_back);
    }
  } else if( is_solid_nghbr(fnode_enum, back_l) ) {
    // der_g = [g(+1) + 3*g(0) - 4*g(-0.5)]/3
      double ux_forw = vx->get_val_ijk(ijk_forw),
        uy_forw = vy->get_val_ijk(ijk_forw),
        uz_forw = vz->get_val_ijk(ijk_forw);

      ux_forw *= a_inv_cr;
      uy_forw *= a_inv_cr;
      uz_forw *= a_inv_cr;

    (*der_ux) = (1.0/3.0)*(ux_forw + 3.0*ux - 4.0*0.0);
    (*der_uy) = (1.0/3.0)*(uy_forw + 3.0*uy - 4.0*0.0);
    (*der_uz) = (1.0/3.0)*(uz_forw + 3.0*uz - 4.0*0.0);
  } else {
    // der_g = [g(+1) - g(-1)]/(2*1)
    double ux_back = vx->get_val_ijk(ijk_back),
      uy_back = vy->get_val_ijk(ijk_back),
      uz_back = vz->get_val_ijk(ijk_back),
      ux_forw = vx->get_val_ijk(ijk_forw),
      uy_forw = vy->get_val_ijk(ijk_forw),
      uz_forw = vz->get_val_ijk(ijk_forw);

    ux_back *= a_inv_cr;
    uy_back *= a_inv_cr;
    uz_back *= a_inv_cr;
    ux_forw *= a_inv_cr;
    uy_forw *= a_inv_cr;
    uz_forw *= a_inv_cr;

    (*der_ux) = 0.5*(ux_forw - ux_back);
    (*der_uy) = 0.5*(uy_forw - uy_back);
    (*der_uz) = 0.5*(uz_forw - uz_back);
  }
}
//---------------------------------------------------------------------------
void IsothermalKernel::initialize_fi(IsothermalNodeData *fdata)
{
  ULLINT fi_ind[Q];
  double fi[Q], feq[Q], fext[Q], KI2[Q][3][3],
    pab_cff = -2.0*a_coll_oper->get_kvisc_relax_t()*CT2,
    gx = a_dt*a_inv_cr*a_gx, gy = a_dt*a_inv_cr*a_gy,
    gz = a_dt*a_inv_cr*a_gz;
  
  get_kin_proj2(KI2);
  NodeSet *fnodes = fdata->get_nodeset();
  UINT fluid_ncount = fnodes->get_node_count();

  FieldData *den_data = fdata->den(), *vx_data = fdata->velx(),
    *vy_data = fdata->vely(), *vz_data = fdata->velz(),
    *fx_data = fdata->frcx(), *fy_data = fdata->frcy(),
    *fz_data = fdata->frcz();
  
  #pragma omp parallel for private(fi_ind, fi, feq, fext)
  for(UINT fnode_enum = 0; fnode_enum < fluid_ncount; ++fnode_enum)
  {
    UINT ijk[3];
    fnodes->get_ijk(fnode_enum, ijk);
    get_fi_ind_arr(fi_ind, fnode_enum);
        
    double den = den_data->get_val_n(fnode_enum),
      ux = vx_data->get_val_n(fnode_enum), uy = vy_data->get_val_n(fnode_enum), 
      uz = vz_data->get_val_n(fnode_enum), fx = fx_data->get_val_n(fnode_enum),
      fy = fy_data->get_val_n(fnode_enum), fz = fz_data->get_val_n(fnode_enum); 

    den *= a_inv_ref_den; ux *= a_inv_cr; uy *= a_inv_cr; uz *= a_inv_cr;
    fx *= a_inv_bf_unit; fy *= a_inv_bf_unit; fz *= a_inv_bf_unit;
    fx += den*gx; fy += den*gy; fz += den*gz;

    a_frc_func(ux,uy,uz,fx,fy,fz,fext);
    a_eq_func(den,ux,uy,uz,feq);
        
    int per_i_e = ((int)ijk[0] + 1 + (int)_nx)%(int)_nx,
        per_i_w = ((int)ijk[0] - 1 + (int)_nx)%(int)_nx,
        per_j_n = ((int)ijk[1] + 1 + (int)_ny)%(int)_ny,
        per_j_s = ((int)ijk[1] - 1 + (int)_ny)%(int)_ny,
        per_k_t = ((int)ijk[2] + 1 + (int)_nz)%(int)_nz,
        per_k_b = ((int)ijk[2] - 1 + (int)_nz)%(int)_nz;

    UINT ijk_e[3] = {(UINT)per_i_e, ijk[1], ijk[2]},
      ijk_w[3] = {(UINT)per_i_w, ijk[1], ijk[2]},
      ijk_n[3] = {ijk[0], (UINT)per_j_n, ijk[2]},
      ijk_s[3] = {ijk[0], (UINT)per_j_s, ijk[2]},
      ijk_t[3] = {ijk[0], ijk[1], (UINT)per_k_t},
      ijk_b[3] = {ijk[0], ijk[1], (UINT)per_k_b};
      
    double dx_ux, dy_ux, dz_ux, dx_uy, dy_uy, dz_uy, dx_uz, dy_uz, dz_uz;
    comp_der(fdata, fnode_enum, E, ijk_e, ijk_w, ux, uy, uz, &dx_ux, &dx_uy, &dx_uz);
    comp_der(fdata, fnode_enum, N, ijk_n, ijk_s, ux, uy, uz, &dy_ux, &dy_uy, &dy_uz);
    comp_der(fdata, fnode_enum, T, ijk_t, ijk_b, ux, uy, uz, &dz_ux, &dz_uy, &dz_uz);

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
IsothermalKernel::IsothermalKernel(Geometry *geom, NodeSet *fnodes,
                                   IsothermalFlowParams *params)
{
  init_ci_masks();
  config_evolve(geom, fnodes);

  a_ref_den = params->ref_den; a_dr = params->dr; a_dt = params->dt;
  a_inv_ref_den = 1.0/a_ref_den; a_inv_dr = 1.0/a_dr; a_inv_dt = 1.0/a_dt;

  a_cr = a_dr*a_inv_dt; a_inv_cr = 1.0/a_cr;
  a_bf_unit = a_ref_den*a_inv_dt*a_cr;
  a_inv_bf_unit = 1.0/a_bf_unit;

  if( params->flow_type == STOKES_FLOW ) {a_eq_func = &eq1;}
  else {a_eq_func = &eq2;}

  if( params->external_forcing ) {a_frc_func = &D3Q19::frc2;}
  else {a_frc_func = &D3Q19::frc1;}
    
  switch( params->collision_operator ) {
    case BGK: {
      a_coll_oper = new BGK_D3Q19();
      break;}
    case TRT: {
      a_coll_oper = new TRT_D3Q19();
      break;}
//    case MRT: {
//      a_coll_oper = new MRT_D3Q19();
//      break;}
//    case REG: {
//      a_coll_oper = new REG_D3Q19();
//      break;}
    default: {
      printf("Warning: using default collision operator (TRT)\n");
      a_coll_oper = new TRT_D3Q19();}
  } 
  set_gravity(params->gx, params->gy, params->gz);
  set_kvisc(params->kvisc);
  a_cs = a_cr*CT;
}
//---------------------------------------------------------------------------
IsothermalKernel::~IsothermalKernel()
{
  dealloc_memory();
}
//---------------------------------------------------------------------------
void IsothermalKernel::set_kvisc(double kvisc)
{
  a_kvisc = kvisc;
  double dimless_kvisc = a_inv_dr*a_inv_cr*kvisc;

  a_coll_oper->set_kvisc(dimless_kvisc);  
}
//---------------------------------------------------------------------------
void IsothermalKernel::set_gravity(double gx, double gy, double gz)
{
  a_gx = gx; a_gy = gy; a_gz = gz;
}
//---------------------------------------------------------------------------
void IsothermalKernel::evolve(IsothermalNodeData *fdata)
{
  NodeSet *fnodes = fdata->get_nodeset();
  FieldData *den_data = fdata->den(), *vx_data = fdata->velx(),
    *vy_data = fdata->vely(), *vz_data = fdata->velz(),
    *fx_data = fdata->frcx(), *fy_data = fdata->frcy(),
    *fz_data = fdata->frcz();

  ULLINT fi_ind[Q];
  UINT fluid_ncount = fnodes->get_node_count();
  double fi[Q], feq[Q], fext[Q], gx = a_dt*a_inv_cr*a_gx,
    gy = a_dt*a_inv_cr*a_gy, gz = a_dt*a_inv_cr*a_gz;

  FI_IND_FPTR fi_ind_func = &IsothermalKernel::get_fi_ind_arr_even_t;
  if( a_odd_t ) fi_ind_func = &IsothermalKernel::get_fi_ind_arr_odd_t;

  #pragma omp parallel for private(fi_ind, fi, feq, fext)
  for(UINT fnode_enum = 0; fnode_enum < fluid_ncount; ++fnode_enum)
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

    double den, ux, uy, uz, inv_den = den_vel_moms(fi, &den, &ux, &uy, &uz),
      fx = fx_data->get_val_n(fnode_enum), fy = fy_data->get_val_n(fnode_enum), 
      fz = fz_data->get_val_n(fnode_enum); 
    
    fx *= a_inv_bf_unit; fy *= a_inv_bf_unit; fz *= a_inv_bf_unit;
    fx += den*gx; fy += den*gy; fz += den*gz;

    ux += 0.5*inv_den*fx; uy += 0.5*inv_den*fy; uz += 0.5*inv_den*fz;
    a_frc_func(ux,uy,uz,fx,fy,fz,fext);
    a_eq_func(den,ux,uy,uz,feq);

    den *= a_ref_den; ux *= a_cr; uy *= a_cr; uz *= a_cr;

    den_data->set_val_n(fnode_enum, den); 
    vx_data->set_val_n(fnode_enum, ux); 
    vy_data->set_val_n(fnode_enum, uy); 
    vz_data->set_val_n(fnode_enum, uz); 

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