//===========================================================================
// Author: Keijo Mattila, JYU, February 2015
// Description: Implementation of LBM collision operators
// Details: 
//===========================================================================
#include <stdio.h>
#include "D3Q19.h"
#include "collision.h"
//===========================================================================
// BGK
//===========================================================================
BGK_D3Q19::BGK_D3Q19()
{
  set_kvisc(1.0/6.0);
}
//---------------------------------------------------------------------------
BGK_D3Q19::~BGK_D3Q19()
{
}
//---------------------------------------------------------------------------
void BGK_D3Q19::set_kvisc(double dimless_kvisc)
{
  using namespace D3Q19;
  BaseCollisionOperator::set_kvisc(dimless_kvisc);

  a_tau = 0.5 + INV_CT2*a_dimless_kvisc;
  a_inv_tau = 1.0/a_tau;
  
  a_rprm_ext = (1.0 - 0.5*a_inv_tau);
  
  if( a_tau > 1.5 ) printf("Warning: dimensionless tau > 1.5!\n");
}                             
//---------------------------------------------------------------------------
void BGK_D3Q19::relax(double *fi, double *feq, double *fext)
{
  using namespace D3Q19;

  fi[BS] = fi[BS] - a_inv_tau*(fi[BS] - feq[BS]) + a_rprm_ext*fext[BS];
  fi[BW] = fi[BW] - a_inv_tau*(fi[BW] - feq[BW]) + a_rprm_ext*fext[BW];
  fi[B]  = fi[B]  - a_inv_tau*(fi[B]  - feq[B])  + a_rprm_ext*fext[B];
  fi[BE] = fi[BE] - a_inv_tau*(fi[BE] - feq[BE]) + a_rprm_ext*fext[BE];
  fi[BN] = fi[BN] - a_inv_tau*(fi[BN] - feq[BN]) + a_rprm_ext*fext[BN];

  fi[SW] = fi[SW] - a_inv_tau*(fi[SW] - feq[SW]) + a_rprm_ext*fext[SW];
  fi[S]  = fi[S]  - a_inv_tau*(fi[S]  - feq[S])  + a_rprm_ext*fext[S];
  fi[SE] = fi[SE] - a_inv_tau*(fi[SE] - feq[SE]) + a_rprm_ext*fext[SE];

  fi[W] = fi[W] - a_inv_tau*(fi[W] - feq[W]) + a_rprm_ext*fext[W];
  fi[C] = fi[C] - a_inv_tau*(fi[C] - feq[C]) + a_rprm_ext*fext[C];
  fi[E] = fi[E] - a_inv_tau*(fi[E] - feq[E]) + a_rprm_ext*fext[E];

  fi[NW] = fi[NW] - a_inv_tau*(fi[NW] - feq[NW]) + a_rprm_ext*fext[NW];
  fi[N]  = fi[N]  - a_inv_tau*(fi[N]  - feq[N])  + a_rprm_ext*fext[N];
  fi[NE] = fi[NE] - a_inv_tau*(fi[NE] - feq[NE]) + a_rprm_ext*fext[NE];

  fi[TS] = fi[TS] - a_inv_tau*(fi[TS] - feq[TS]) + a_rprm_ext*fext[TS];
  fi[TW] = fi[TW] - a_inv_tau*(fi[TW] - feq[TW]) + a_rprm_ext*fext[TW];
  fi[T]  = fi[T]  - a_inv_tau*(fi[T]  - feq[T])  + a_rprm_ext*fext[T];
  fi[TE] = fi[TE] - a_inv_tau*(fi[TE] - feq[TE]) + a_rprm_ext*fext[TE];
  fi[TN] = fi[TN] - a_inv_tau*(fi[TN] - feq[TN]) + a_rprm_ext*fext[TN];
}                             
//===========================================================================
// TRT
//===========================================================================
TRT_D3Q19::TRT_D3Q19()
{
  set_kvisc(1.0/6.0);
}
//---------------------------------------------------------------------------
TRT_D3Q19::~TRT_D3Q19()
{
}
//---------------------------------------------------------------------------
void TRT_D3Q19::set_kvisc(double dimless_kvisc)
{
  using namespace D3Q19;
  BaseCollisionOperator::set_kvisc(dimless_kvisc);

  a_tau_e = 0.5 + INV_CT2*a_dimless_kvisc;
  a_inv_tau_e = 1.0/a_tau_e;
  a_inv_tau_o = 8.0*((2.0 - a_inv_tau_e)/(8.0 - a_inv_tau_e));
  
  a_rprm_ext_e = (1.0 - 0.5*a_inv_tau_e);
  a_rprm_ext_o = (1.0 - 0.5*a_inv_tau_o);
  
  if( a_tau_e > 1.5 ) printf("Warning: dimensionless tau > 1.5!\n");
}                             
//---------------------------------------------------------------------------
void TRT_D3Q19::relax(double *fi, double *feq, double *fext)
{
  using namespace D3Q19;

  //-----------------------------------------------
  // BS and TN
  //-----------------------------------------------
  double fneq_bs = fi[BS] - feq[BS],
    fneq_opp_bs  = fi[RDIR[BS]] - feq[RDIR[BS]],
    fneq_even_bs = 0.5*(fneq_bs + fneq_opp_bs),
    fneq_odd_bs  = 0.5*(fneq_bs - fneq_opp_bs),
    fext_even_bs = 0.5*(fext[BS] + fext[RDIR[BS]]),
    fext_odd_bs  = 0.5*(fext[BS] - fext[RDIR[BS]]),
    rlx_even_bs = a_inv_tau_e*fneq_even_bs - a_rprm_ext_e*fext_even_bs,
    rlx_odd_bs  = a_inv_tau_o*fneq_odd_bs - a_rprm_ext_o*fext_odd_bs;
      
  fi[BS] = fi[BS] - rlx_even_bs - rlx_odd_bs;
  fi[RDIR[BS]] = fi[RDIR[BS]] - rlx_even_bs + rlx_odd_bs;

  //-----------------------------------------------
  // BW and TE
  //-----------------------------------------------
  double fneq_bw = fi[BW] - feq[BW],
    fneq_opp_bw  = fi[RDIR[BW]] - feq[RDIR[BW]],
    fneq_even_bw = 0.5*(fneq_bw + fneq_opp_bw),
    fneq_odd_bw  = 0.5*(fneq_bw - fneq_opp_bw),
    fext_even_bw = 0.5*(fext[BW] + fext[RDIR[BW]]),
    fext_odd_bw  = 0.5*(fext[BW] - fext[RDIR[BW]]),
    rlx_even_bw = a_inv_tau_e*fneq_even_bw - a_rprm_ext_e*fext_even_bw,
    rlx_odd_bw  = a_inv_tau_o*fneq_odd_bw - a_rprm_ext_o*fext_odd_bw;
      
  fi[BW] = fi[BW] - rlx_even_bw - rlx_odd_bw;
  fi[RDIR[BW]] = fi[RDIR[BW]] - rlx_even_bw + rlx_odd_bw;

  //-----------------------------------------------
  // B and T
  //-----------------------------------------------
  double fneq_b = fi[B] - feq[B],
    fneq_opp_b  = fi[RDIR[B]] - feq[RDIR[B]],
    fneq_even_b = 0.5*(fneq_b + fneq_opp_b),
    fneq_odd_b  = 0.5*(fneq_b - fneq_opp_b),
    fext_even_b = 0.5*(fext[B] + fext[RDIR[B]]),
    fext_odd_b  = 0.5*(fext[B] - fext[RDIR[B]]),
    rlx_even_b = a_inv_tau_e*fneq_even_b - a_rprm_ext_e*fext_even_b,
    rlx_odd_b  = a_inv_tau_o*fneq_odd_b - a_rprm_ext_o*fext_odd_b;
      
  fi[B] = fi[B] - rlx_even_b - rlx_odd_b;
  fi[RDIR[B]] = fi[RDIR[B]] - rlx_even_b + rlx_odd_b;
  
  //-----------------------------------------------
  // BE and TW
  //-----------------------------------------------
  double fneq_be = fi[BE] - feq[BE],
    fneq_opp_be  = fi[RDIR[BE]] - feq[RDIR[BE]],
    fneq_even_be = 0.5*(fneq_be + fneq_opp_be),
    fneq_odd_be  = 0.5*(fneq_be - fneq_opp_be),
    fext_even_be = 0.5*(fext[BE] + fext[RDIR[BE]]),
    fext_odd_be  = 0.5*(fext[BE] - fext[RDIR[BE]]),
    rlx_even_be = a_inv_tau_e*fneq_even_be - a_rprm_ext_e*fext_even_be,
    rlx_odd_be  = a_inv_tau_o*fneq_odd_be - a_rprm_ext_o*fext_odd_be;
      
  fi[BE] = fi[BE] - rlx_even_be - rlx_odd_be;
  fi[RDIR[BE]] = fi[RDIR[BE]] - rlx_even_be + rlx_odd_be;
  
  //-----------------------------------------------
  // BN and TS
  //-----------------------------------------------
  double fneq_bn = fi[BN] - feq[BN],
    fneq_opp_bn  = fi[RDIR[BN]] - feq[RDIR[BN]],
    fneq_even_bn = 0.5*(fneq_bn + fneq_opp_bn),
    fneq_odd_bn  = 0.5*(fneq_bn - fneq_opp_bn),
    fext_even_bn = 0.5*(fext[BN] + fext[RDIR[BN]]),
    fext_odd_bn  = 0.5*(fext[BN] - fext[RDIR[BN]]),
    rlx_even_bn = a_inv_tau_e*fneq_even_bn - a_rprm_ext_e*fext_even_bn,
    rlx_odd_bn  = a_inv_tau_o*fneq_odd_bn - a_rprm_ext_o*fext_odd_bn;
      
  fi[BN] = fi[BN] - rlx_even_bn - rlx_odd_bn;
  fi[RDIR[BN]] = fi[RDIR[BN]] - rlx_even_bn + rlx_odd_bn;

  //-----------------------------------------------
  // SW and NE
  //-----------------------------------------------
  double fneq_sw = fi[SW] - feq[SW],
    fneq_opp_sw  = fi[RDIR[SW]] - feq[RDIR[SW]],
    fneq_even_sw = 0.5*(fneq_sw + fneq_opp_sw),
    fneq_odd_sw  = 0.5*(fneq_sw - fneq_opp_sw),
    fext_even_sw = 0.5*(fext[SW] + fext[RDIR[SW]]),
    fext_odd_sw  = 0.5*(fext[SW] - fext[RDIR[SW]]),
    rlx_even_sw = a_inv_tau_e*fneq_even_sw - a_rprm_ext_e*fext_even_sw,
    rlx_odd_sw  = a_inv_tau_o*fneq_odd_sw - a_rprm_ext_o*fext_odd_sw;
      
  fi[SW] = fi[SW] - rlx_even_sw - rlx_odd_sw;
  fi[RDIR[SW]] = fi[RDIR[SW]] - rlx_even_sw + rlx_odd_sw;

  //-----------------------------------------------
  // S and N
  //-----------------------------------------------
  double fneq_s = fi[S] - feq[S],
    fneq_opp_s  = fi[RDIR[S]] - feq[RDIR[S]],
    fneq_even_s = 0.5*(fneq_s + fneq_opp_s),
    fneq_odd_s  = 0.5*(fneq_s - fneq_opp_s),
    fext_even_s = 0.5*(fext[S] + fext[RDIR[S]]),
    fext_odd_s  = 0.5*(fext[S] - fext[RDIR[S]]),
    rlx_even_s = a_inv_tau_e*fneq_even_s - a_rprm_ext_e*fext_even_s,
    rlx_odd_s  = a_inv_tau_o*fneq_odd_s - a_rprm_ext_o*fext_odd_s;
      
  fi[S] = fi[S] - rlx_even_s - rlx_odd_s;
  fi[RDIR[S]] = fi[RDIR[S]] - rlx_even_s + rlx_odd_s;

  //-----------------------------------------------
  // SE and NW
  //-----------------------------------------------
  double fneq_se = fi[SE] - feq[SE],
    fneq_opp_se  = fi[RDIR[SE]] - feq[RDIR[SE]],
    fneq_even_se = 0.5*(fneq_se + fneq_opp_se),
    fneq_odd_se  = 0.5*(fneq_se - fneq_opp_se),
    fext_even_se = 0.5*(fext[SE] + fext[RDIR[SE]]),
    fext_odd_se  = 0.5*(fext[SE] - fext[RDIR[SE]]),
    rlx_even_se = a_inv_tau_e*fneq_even_se - a_rprm_ext_e*fext_even_se,
    rlx_odd_se  = a_inv_tau_o*fneq_odd_se - a_rprm_ext_o*fext_odd_se;
      
  fi[SE] = fi[SE] - rlx_even_se - rlx_odd_se;
  fi[RDIR[SE]] = fi[RDIR[SE]] - rlx_even_se + rlx_odd_se;

  //-----------------------------------------------
  // W and E
  //-----------------------------------------------
  double fneq_w = fi[W] - feq[W],
    fneq_opp_w  = fi[RDIR[W]] - feq[RDIR[W]],
    fneq_even_w = 0.5*(fneq_w + fneq_opp_w),
    fneq_odd_w  = 0.5*(fneq_w - fneq_opp_w),
    fext_even_w = 0.5*(fext[W] + fext[RDIR[W]]),
    fext_odd_w  = 0.5*(fext[W] - fext[RDIR[W]]),
    rlx_even_w = a_inv_tau_e*fneq_even_w - a_rprm_ext_e*fext_even_w,
    rlx_odd_w  = a_inv_tau_o*fneq_odd_w - a_rprm_ext_o*fext_odd_w;
      
  fi[W] = fi[W] - rlx_even_w - rlx_odd_w;
  fi[RDIR[W]] = fi[RDIR[W]] - rlx_even_w + rlx_odd_w;

  //-----------------------------------------------
  // C
  //-----------------------------------------------
  fi[C] = fi[C] - a_inv_tau_e*(fi[C] - feq[C]) + a_rprm_ext_e*fext[C];
}
//===========================================================================