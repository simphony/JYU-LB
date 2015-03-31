//===========================================================================
// Author: Keijo Mattila, JYU, February 2015
// Description: Implementation of LBM solvers
// Details: Defines data structures for flow parameters and state information
//===========================================================================
#include <math.h>
#include <stdio.h>
#include "solver.h"
//===========================================================================
IsothermalSolver::IsothermalSolver(Geometry *geom,
                                   IsothermalFlowParams *params)
{
  _lat = new Lattice(geom->get_lattice());
  
  GeomDataFilter fnode_filter(geom->get_phase(), FLUID_NODE);
  _fnodes = new NodeSubset(_lat, &fnode_filter);

  double iden = params->ref_den, ivel[3] = {0,0,0}, ifrc[3] = {0,0,0};
  _fdata = new IsothermalNodeData(_fnodes, iden, ivel, ifrc);

  _kernel = new IsothermalKernel(geom, _fnodes, params);
  _is_initialized = false;
}
//---------------------------------------------------------------------------
IsothermalSolver::~IsothermalSolver()
{
  if( _lat != 0 ) {
    delete _lat;
    _lat = 0;
  }
  if( _fnodes != 0 ) {
    delete _fnodes;
    _fnodes = 0;
  }
  if( _fdata != 0 ) {
    delete _fdata;
    _fdata = 0;
  }
  if( _kernel != 0 ) {
    delete _kernel;
    _kernel = 0;
  }
}
//---------------------------------------------------------------------------
void IsothermalSolver::init_field_data()  throw(std::runtime_error)
{
  if( _is_initialized ) {
    throw std::runtime_error("Field variables already initialized!");
  }
  _kernel->initialize_fi(_fdata);
  _is_initialized = true;
}
//---------------------------------------------------------------------------
void IsothermalSolver::evolve(UINT tsteps) throw(std::runtime_error)
{
  if( !_is_initialized ) {
    throw std::runtime_error("Field variables not initialized!");
  }
  for(UINT t = 0; t < tsteps; ++t) {
    _kernel->evolve(_fdata);
  }
}
//---------------------------------------------------------------------------
void IsothermalSolver::calc_flow_info(IsothermalFlowInfo *info)
{
  NodeSet *fnodes = _fdata->get_nodeset();
  UINT fluid_ncount = fnodes->get_node_count();

  double rden = _kernel->get_ref_den(),
    pri_max_den, pri_min_den, pri_max_u2, 
    sha_max_den = rden, sha_min_den = rden, sha_max_u2 = 0.0, 
    tot_den = 0.0, tot_ux = 0.0, tot_uy = 0.0, tot_uz = 0.0, tot_u2 = 0.0,
    tot_jx = 0.0, tot_jy = 0.0, tot_jz = 0.0, tot_j2 = 0.0;

  #pragma omp parallel private(pri_max_den,pri_min_den,pri_max_u2)
  {
    pri_max_den = rden;
    pri_min_den = rden; 
    pri_max_u2 = 0.0; 

    #pragma omp for reduction(+:tot_den,tot_ux,tot_uy,tot_uz,tot_u2,tot_jx,tot_jy,tot_jz,tot_j2)
    for(UINT fnode_enum = 0; fnode_enum < fluid_ncount; ++fnode_enum)
    {
      double den = _fdata->den()->get_val_n(fnode_enum),
        ux = _fdata->velx()->get_val_n(fnode_enum),
        uy = _fdata->vely()->get_val_n(fnode_enum),
        uz = _fdata->velz()->get_val_n(fnode_enum),
      
        jx = den*ux, jy = den*uy, jz = den*uz,
        u2 = sqrt(ux*ux + uy*uy + uz*uz),
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