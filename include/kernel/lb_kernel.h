//===========================================================================
// Author: Keijo Mattila, JYU, February 2015
// Description: Implementation of LBM kernels
// Details: Computation in dimensionless variables,
//          input/output with dimensional variables
//===========================================================================
#ifndef LB_KERNEL_H
#define LB_KERNEL_H
//===========================================================================
#include "lattice.h"
#include "lb_collision.h"
//===========================================================================
namespace LB_Kernel
{
  typedef unsigned long long int ULLINT;

  typedef void (*EQ3D_FPTR)(double,double,double,double,double *);
  typedef void (*FRC3D_FPTR)(double,double,double,
                               double,double,double,double *);
                               
  class Isothermal3D
  {
    public:
      Isothermal3D(Lattice::Isothermal3D *lat, double ref_den, double dr,
        double dt, double kvisc, double gx, double gy, double gz,
        unsigned char flow_type, unsigned char coll_oper,
        bool external_forcing);
      ~Isothermal3D();
     
      void set_kvisc(double kvisc);
      void set_gravity(double gx, double gy, double gz);
      void evolve(Lattice::Isothermal3D *lat);

      //-------------------------------------------------------------------
      // Get methods
      //-------------------------------------------------------------------
      double get_ref_den() const {return a_ref_den;}
      double get_dr() const {return a_dr;}
      double get_dt() const {return a_dt;}
      double get_kvisc() const {return a_kvisc;} // kinematic viscosity
      double get_cs() const {return a_cs;} // speed of sound
      void get_gravity(double *gx, double *gy, double *gz) const {
        (*gx) = a_gx; (*gy) = a_gy; (*gz) = a_gz;
      }
    private:
      double *a_fi;
      ULLINT *a_mem_addr;
      LB_Collision::BaseOperator *a_coll_oper;

      EQ3D_FPTR a_eq_func;
      FRC3D_FPTR a_frc_func;
      
      // Variables for dimensionless <-> dimensional conversion
      double a_ref_den, a_dr, a_dt, a_cr, a_kvisc, a_cs, a_gx, a_gy, a_gz;
      double a_inv_ref_den, a_inv_dr, a_inv_dt, a_inv_cr;
      double a_bf_unit, a_inv_bf_unit;
      
      bool a_odd_t; // Aux.variable for the AA-pattern algorithm
      ULLINT a_ull_fluid_ncount; // Aux.variable for memory accessing
      
      void get_fi_ind_arr_odd_t(ULLINT *fi_ind, ULLINT fnode_enum);
      void get_fi_ind_arr_even_t(ULLINT *fi_ind, ULLINT fnode_enum);
      void get_fi_ind_arr(ULLINT *fi_ind, ULLINT fnode_enum);
      ULLINT get_fi_ind(ULLINT fnode_enum, unsigned char l);

      void alloc_memory(unsigned int fluid_ncount);
      void dealloc_memory();
      
      void config_evolve(Lattice::Isothermal3D *lat);
      void initialize_fi(Lattice::Isothermal3D *lat);
      
      void comp_der(Lattice::Isothermal3D *lat, unsigned int back_nn,
        unsigned int forw_nn, double ux, double uy, double uz,
        double *der_ux, double *der_uy, double *der_uz);
  };
}
//===========================================================================
#endif
//===========================================================================