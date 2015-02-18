//===========================================================================
// Author: Keijo Mattila, JYU, February 2015
// Description: Implementation of LBM solvers
// Details: Defines data structures for flow parameters and state information
//===========================================================================
#include <math.h>
#include "lb_solver.h"
//===========================================================================
using namespace LB_Solver::Isothermal3D;
//===========================================================================
TimeStepper::TimeStepper(Lattice::Isothermal3D *lat,FlowParams *params)
{
  a_kernel = new LB_Kernel::Isothermal3D(lat, params->ref_den, params->dr,
    params->dt, params->kvisc, params->gx, params->gy, params->gz,
    params->flow_type, params->collision_operator, params->external_forcing);
}
//---------------------------------------------------------------------------
TimeStepper::~TimeStepper()
{
  if( a_kernel != 0 ) {
    delete a_kernel;
    a_kernel = 0;
  }
}
//---------------------------------------------------------------------------
void TimeStepper::evolve(Lattice::Isothermal3D *lat, unsigned int tsteps)
{
  for(unsigned int t = 0; t < tsteps; ++t) {
    a_kernel->evolve(lat);
  }
}
//---------------------------------------------------------------------------
void TimeStepper::calc_flow_info(Lattice::Isothermal3D *lat, FlowInfo *info)
{
  double rden = a_kernel->get_ref_den(),
    pri_max_den, pri_min_den, pri_max_u2, 
    sha_max_den = rden, sha_min_den = rden, sha_max_u2 = 0.0, 
    tot_den = 0.0, tot_ux = 0.0, tot_uy = 0.0, tot_uz = 0.0, tot_u2 = 0.0,
    tot_jx = 0.0, tot_jy = 0.0, tot_jz = 0.0, tot_j2 = 0.0;

  unsigned int fluid_ncount = lat->fluid_ncount();

  #pragma omp parallel private(pri_max_den,pri_min_den,pri_max_u2)
  {
    pri_max_den = rden;
    pri_min_den = rden; 
    pri_max_u2 = 0.0; 

    #pragma omp for reduction(+:tot_den,tot_ux,tot_uy,tot_uz,tot_u2,tot_jx,tot_jy,tot_jz,tot_j2)
    for(unsigned int fnode_enum = 0; fnode_enum < fluid_ncount; ++fnode_enum)
    {
      double den, ux, uy, uz, u2, jx, jy, jz, j2;
      lat->get_fnode_den(fnode_enum, &den);
      lat->get_fnode_vel(fnode_enum, &ux, &uy, &uz);

      jx = den*ux; jy = den*uy; jz = den*uz;
      u2 = sqrt(ux*ux + uy*uy + uz*uz);
      j2 = den*u2;
      
      tot_den += den;
 	    tot_ux += ux; tot_uy += uy; tot_uz += uz; tot_u2 += u2;
 	    tot_jx += jx; tot_jy += jy; tot_jz += jz; tot_j2 += j2;

      if( den > pri_max_den ) pri_max_den = den;
      if( den < pri_min_den ) pri_min_den = den;
      if( u2 > pri_max_u2 )   pri_max_u2 = u2;
    }
    #pragma omp critical
    {
      if( pri_max_den > sha_max_den ) sha_max_den = pri_max_den;
      if( pri_min_den < sha_min_den ) sha_min_den = pri_min_den;
      if( pri_max_u2 > sha_max_u2 )   sha_max_u2 = pri_max_u2;
    }
  }
  info->tot_jx = tot_jx;
  info->tot_jy = tot_jy;
  info->tot_jz = tot_jz;
  info->tot_j2 = tot_j2;
  info->tot_den = tot_den;

  if( fluid_ncount > 0 ) {
    double ave_cff = 1.0/(double)fluid_ncount;
    info->ave_ux = ave_cff*tot_ux;
    info->ave_uy = ave_cff*tot_uy;
    info->ave_uz = ave_cff*tot_uz;
    info->ave_u2 = ave_cff*tot_u2;
  } else {
    info->ave_ux = 0.0;
    info->ave_uy = 0.0;
    info->ave_uz = 0.0;
    info->ave_u2 = 0.0;
  }
  info->max_den = sha_max_den;
  info->min_den = sha_min_den;
  info->max_u2 = sha_max_u2;
}
//===========================================================================