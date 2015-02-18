//===========================================================================
// Author: Keijo Mattila, JYU, February 2015
// Description: Implementation of LBM solvers
// Details: Defines data structures for flow parameters and state information
//===========================================================================
#ifndef LB_SOLVER_H
#define LB_SOLVER_H
//===========================================================================
#include "lb_kernel.h"
//===========================================================================
namespace LB_Solver
{
  namespace Isothermal3D
  {
    struct FlowParams
    {
      double dr, dt;
      double ref_den, kvisc, gx, gy, gz;
      unsigned char flow_type, collision_operator;
      bool external_forcing;
    };

    struct FlowInfo 
    {
      double tot_jx, tot_jy, tot_jz, tot_j2, tot_den;
      double ave_ux, ave_uy, ave_uz, ave_u2;
      double max_u2, max_den, min_den;
    };
  
    class TimeStepper
    {
      public:
        TimeStepper(Lattice::Isothermal3D *lat, FlowParams *params);
        ~TimeStepper();
      
        void set_kvisc(double kvisc) {a_kernel->set_kvisc(kvisc);}
        void set_gravity(double gx, double gy, double gz) {
          a_kernel->set_gravity(gx, gy, gz);
        }
        void evolve(Lattice::Isothermal3D *lat, unsigned int tsteps);
        void calc_flow_info(Lattice::Isothermal3D *lat, FlowInfo *info);
        //-------------------------------------------------------------------
        // Get methods
        //-------------------------------------------------------------------
        double get_ref_den() const {return a_kernel->get_ref_den();}
        double get_dr() const {return a_kernel->get_dr();}
        double get_dt() const {return a_kernel->get_dt();}
        double get_kvisc() const {return a_kernel->get_kvisc();}
        double get_cs() const {return a_kernel->get_cs();}

      private:
        LB_Kernel::Isothermal3D *a_kernel;
    };
  }
}
//===========================================================================
#endif
//===========================================================================