//===========================================================================
// Author: Keijo Mattila, JYU, February 2015
// Description: Implementation of LBM kernels
// Details: Computation in dimensionless variables,
//          input/output with dimensional variables
//===========================================================================
#ifndef KERNEL_H
#define KERNEL_H
//===========================================================================
#include "data.h"
#include "collision.h"
//===========================================================================
struct IsothermalFlowParams
{
  double dr, dt;
  double ref_den, kvisc, gx, gy, gz;
  unsigned char flow_type, collision_operator;
  bool external_forcing;
};
//--------------------------------------------------------------------------- 
class IsothermalKernel
{
  public:
    IsothermalKernel(Geometry *geom, NodeSet *fnodes,
                     IsothermalFlowParams *params);
    ~IsothermalKernel();
     
    void set_kvisc(double kvisc);
    void set_gravity(double gx, double gy, double gz);

    void initialize_fi(IsothermalNodeData *fdata);
    void evolve(IsothermalNodeData *fdata);

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
    UINT _nx, _ny, _nz, *_nghbr_info;
    BaseCollisionOperator *a_coll_oper;

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

    void alloc_memory(UINT fluid_ncount);
    void dealloc_memory();
      
    void config_evolve(Geometry *geom, NodeSet *fnodes);
      
    void comp_der(IsothermalNodeData *fdata, UINT fnode_enum, 
      unsigned char forw_l, UINT ijk_forw[3], UINT ijk_back[3],
      double ux, double uy, double uz, double *der_ux,
      double *der_uy, double *der_uz);
        
    bool is_solid_nghbr(UINT fnode_enum, unsigned char l);
    UINT get_solid_nghbr_count(UINT fnode_enum);
};
//===========================================================================
#endif
//===========================================================================