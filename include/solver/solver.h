//===========================================================================
// Author: Keijo Mattila, JYU, February 2015
// Description: Implementation of LBM solvers
// Details: Defines data structures for flow parameters and state information
//===========================================================================
#ifndef SOLVER_H
#define SOLVER_H
//===========================================================================
#include "kernel.h"
//===========================================================================
struct IsothermalFlowInfo 
{
  double tot_jx, tot_jy, tot_jz, tot_j2, tot_den;
  double ave_ux, ave_uy, ave_uz, ave_u2;
  double max_u2, max_den, min_den;
};
//--------------------------------------------------------------------------- 
class IsothermalSolver
{
  public:
    IsothermalSolver(Geometry *geom, IsothermalFlowParams *params);
    ~IsothermalSolver();
      
    void set_kvisc(double kvisc) {_kernel->set_kvisc(kvisc);}
    void set_gravity(double gx, double gy, double gz) {
      _kernel->set_gravity(gx, gy, gz);
    }
    void init_field_data() throw(std::runtime_error);
    void evolve(UINT tsteps) throw(std::runtime_error);

    void calc_flow_info(IsothermalFlowInfo *info);
    //-----------------------------------------------------------------------
    // Get methods
    //-----------------------------------------------------------------------
    double get_ref_den() const {return _kernel->get_ref_den();}
    double get_dr() const {return _kernel->get_dr();}
    double get_dt() const {return _kernel->get_dt();}
    double get_kvisc() const {return _kernel->get_kvisc();}
    double get_cs() const {return _kernel->get_cs();}
    IsothermalNodeData *get_fluid_node_data() const {return _fdata;}

    private:
      Lattice *_lat;
      NodeSet *_fnodes;
      IsothermalKernel *_kernel;
      IsothermalNodeData *_fdata;
      bool _is_initialized;
};
//===========================================================================
#endif
//===========================================================================