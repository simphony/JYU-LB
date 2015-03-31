//===========================================================================
// Author: Keijo Mattila, JYU, February 2015
// Description: Implementation of LBM collision operators
// Details: 
//===========================================================================
#ifndef COLLISION_H
#define COLLISION_H
//===========================================================================
// Base class
//-------------------------------------------------------------------------
class BaseCollisionOperator
{
  public:
    BaseCollisionOperator() {}
    virtual ~BaseCollisionOperator() {}
      
    virtual void set_kvisc(double dimless_kvisc) {
      a_dimless_kvisc = dimless_kvisc;
    }
    virtual void relax(double *fi, double *feq, double *fext) = 0;
    virtual double get_kvisc_relax_t() = 0; // dimensionless
      
  protected:
    double a_dimless_kvisc;
};
//-------------------------------------------------------------------------
// BGK for the D3Q19 velocity set
//-------------------------------------------------------------------------
class BGK_D3Q19: public BaseCollisionOperator
{
  public:
    BGK_D3Q19();
    ~BGK_D3Q19();
      
    void set_kvisc(double dimless_kvisc);
    void relax(double *fi, double *feq, double *fext);
    double get_kvisc_relax_t() {return a_tau;} // dimensionless

  private:
    double a_tau, a_inv_tau, a_rprm_ext;
};
//-------------------------------------------------------------------------
// TRT for the D3Q19 velocity set
//-------------------------------------------------------------------------
class TRT_D3Q19: public BaseCollisionOperator
{
  public:
    TRT_D3Q19();
    ~TRT_D3Q19();
    
    void set_kvisc(double dimless_kvisc);
    void relax(double *fi, double *feq, double *fext);
    double get_kvisc_relax_t() {return a_tau_e;} // dimensionless

  private:
    double a_tau_e, a_inv_tau_e, a_inv_tau_o, a_rprm_ext_e, a_rprm_ext_o;
};
//===========================================================================
#endif
//===========================================================================