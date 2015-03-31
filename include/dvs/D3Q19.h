//===========================================================================
// Author: Keijo Mattila, JYU, February 2015
// Description: Specification of the D3Q19 discrete velocity set
// Details: Isothermal equilibrium and forcing functions up to 2nd-order
//===========================================================================
#ifndef D3Q19_H
#define D3Q19_H
//===========================================================================
#include <math.h>
#include "def.h"
#include "const.h"
//===========================================================================
// Five alternative data layouts for the distribution values of D3Q19
//---------------------------------------------------------------------------
// 1. Collision optimized data layout
#ifdef COLL_OPT_DATA_LAYOUT
  #define F_BS_IND(fn_enum,fn_count) (19ull*(fn_enum))
  #define F_BW_IND(fn_enum,fn_count) (19ull*(fn_enum) + 1ull)
  #define F_B_IND(fn_enum,fn_count)  (19ull*(fn_enum) + 2ull)
  #define F_BE_IND(fn_enum,fn_count) (19ull*(fn_enum) + 3ull)
  #define F_BN_IND(fn_enum,fn_count) (19ull*(fn_enum) + 4ull)
  #define F_SW_IND(fn_enum,fn_count) (19ull*(fn_enum) + 5ull)
  #define F_S_IND(fn_enum,fn_count)  (19ull*(fn_enum) + 6ull)
  #define F_SE_IND(fn_enum,fn_count) (19ull*(fn_enum) + 7ull)
  #define F_W_IND(fn_enum,fn_count)  (19ull*(fn_enum) + 8ull)
  #define F_C_IND(fn_enum,fn_count)  (19ull*(fn_enum) + 9ull)
  #define F_E_IND(fn_enum,fn_count)  (19ull*(fn_enum) + 10ull)
  #define F_NW_IND(fn_enum,fn_count) (19ull*(fn_enum) + 11ull)
  #define F_N_IND(fn_enum,fn_count)  (19ull*(fn_enum) + 12ull)
  #define F_NE_IND(fn_enum,fn_count) (19ull*(fn_enum) + 13ull)
  #define F_TS_IND(fn_enum,fn_count) (19ull*(fn_enum) + 14ull)
  #define F_TW_IND(fn_enum,fn_count) (19ull*(fn_enum) + 15ull)
  #define F_T_IND(fn_enum,fn_count)  (19ull*(fn_enum) + 16ull)
  #define F_TE_IND(fn_enum,fn_count) (19ull*(fn_enum) + 17ull)
  #define F_TN_IND(fn_enum,fn_count) (19ull*(fn_enum) + 18ull)
#endif
//---------------------------------------------------------------------------
// 2. Propagation optimized data layout
#ifdef PROP_OPT_DATA_LAYOUT
  #define F_BS_IND(fn_enum,fn_count) (fn_enum)
  #define F_BW_IND(fn_enum,fn_count) (1ull*(fn_count) + (fn_enum))
  #define F_B_IND(fn_enum,fn_count)  (2ull*(fn_count) + (fn_enum))
  #define F_BE_IND(fn_enum,fn_count) (3ull*(fn_count) + (fn_enum))
  #define F_BN_IND(fn_enum,fn_count) (4ull*(fn_count) + (fn_enum))
  #define F_SW_IND(fn_enum,fn_count) (5ull*(fn_count) + (fn_enum))
  #define F_S_IND(fn_enum,fn_count)  (6ull*(fn_count) + (fn_enum))
  #define F_SE_IND(fn_enum,fn_count) (7ull*(fn_count) + (fn_enum))
  #define F_W_IND(fn_enum,fn_count)  (8ull*(fn_count)  + (fn_enum))
  #define F_C_IND(fn_enum,fn_count)  (9ull*(fn_count)  + (fn_enum))
  #define F_E_IND(fn_enum,fn_count)  (10ull*(fn_count) + (fn_enum))
  #define F_NW_IND(fn_enum,fn_count) (11ull*(fn_count) + (fn_enum))
  #define F_N_IND(fn_enum,fn_count)  (12ull*(fn_count) + (fn_enum))
  #define F_NE_IND(fn_enum,fn_count) (13ull*(fn_count) + (fn_enum))
  #define F_TS_IND(fn_enum,fn_count) (14ull*(fn_count) + (fn_enum))
  #define F_TW_IND(fn_enum,fn_count) (15ull*(fn_count) + (fn_enum))
  #define F_T_IND(fn_enum,fn_count)  (16ull*(fn_count) + (fn_enum))
  #define F_TE_IND(fn_enum,fn_count) (17ull*(fn_count) + (fn_enum))
  #define F_TN_IND(fn_enum,fn_count) (18ull*(fn_count) + (fn_enum))
#endif
//---------------------------------------------------------------------------
// 3. Bundle A data layout
#ifdef BUNDLE_A_DATA_LAYOUT
  #define F_BS_IND(fn_enum,fn_count) (fn_enum)
  #define F_BW_IND(fn_enum,fn_count) ((fn_count)+3ull*(fn_enum))
  #define F_B_IND(fn_enum,fn_count)  ((fn_count)+3ull*(fn_enum)+1ull)
  #define F_BE_IND(fn_enum,fn_count) ((fn_count)+3ull*(fn_enum)+2ull)
  #define F_BN_IND(fn_enum,fn_count) (4ull*(fn_count)+(fn_enum))
  #define F_SW_IND(fn_enum,fn_count) (5ull*(fn_count)+3ull*(fn_enum))
  #define F_S_IND(fn_enum,fn_count)  (5ull*(fn_count)+3ull*(fn_enum)+1ull)
  #define F_SE_IND(fn_enum,fn_count) (5ull*(fn_count)+3ull*(fn_enum)+2ull)
  #define F_W_IND(fn_enum,fn_count)  (8ull*(fn_count)+3ull*(fn_enum))
  #define F_C_IND(fn_enum,fn_count)  (8ull*(fn_count)+3ull*(fn_enum)+1ull)
  #define F_E_IND(fn_enum,fn_count)  (8ull*(fn_count)+3ull*(fn_enum)+2ull)
  #define F_NW_IND(fn_enum,fn_count) (11ull*(fn_count)+3ull*(fn_enum))
  #define F_N_IND(fn_enum,fn_count)  (11ull*(fn_count)+3ull*(fn_enum)+1ull)
  #define F_NE_IND(fn_enum,fn_count) (11ull*(fn_count)+3ull*(fn_enum)+2ull)
  #define F_TS_IND(fn_enum,fn_count) (14ull*(fn_count)+(fn_enum))
  #define F_TW_IND(fn_enum,fn_count) (15ull*(fn_count)+3ull*(fn_enum))
  #define F_T_IND(fn_enum,fn_count)  (15ull*(fn_count)+3ull*(fn_enum)+1ull)
  #define F_TE_IND(fn_enum,fn_count) (15ull*(fn_count)+3ull*(fn_enum)+2ull)
  #define F_TN_IND(fn_enum,fn_count) (18ull*(fn_count)+(fn_enum))
#endif
//---------------------------------------------------------------------------
// 4. Bundle B data layout
#ifdef BUNDLE_B_DATA_LAYOUT
  #define F_BS_IND(fn_enum,fn_count) (5ull*(fn_enum))
  #define F_BW_IND(fn_enum,fn_count) (5ull*(fn_enum)+1ull)
  #define F_B_IND(fn_enum,fn_count)  (5ull*(fn_enum)+2ull)
  #define F_BE_IND(fn_enum,fn_count) (5ull*(fn_enum)+3ull)
  #define F_BN_IND(fn_enum,fn_count) (5ull*(fn_enum)+4ull)
  #define F_SW_IND(fn_enum,fn_count) (5ull*(fn_count)+3ull*(fn_enum))
  #define F_S_IND(fn_enum,fn_count)  (5ull*(fn_count)+3ull*(fn_enum)+1ull)
  #define F_SE_IND(fn_enum,fn_count) (5ull*(fn_count)+3ull*(fn_enum)+2ull)
  #define F_W_IND(fn_enum,fn_count)  (8ull*(fn_count)+3ull*(fn_enum))
  #define F_C_IND(fn_enum,fn_count)  (8ull*(fn_count)+3ull*(fn_enum)+1ull)
  #define F_E_IND(fn_enum,fn_count)  (8ull*(fn_count)+3ull*(fn_enum)+2ull)
  #define F_NW_IND(fn_enum,fn_count) (11ull*(fn_count)+3ull*(fn_enum))
  #define F_N_IND(fn_enum,fn_count)  (11ull*(fn_count)+3ull*(fn_enum)+1ull)
  #define F_NE_IND(fn_enum,fn_count) (11ull*(fn_count)+3ull*(fn_enum)+2ull)
  #define F_TS_IND(fn_enum,fn_count) (14ull*(fn_count)+5ull*(fn_enum))
  #define F_TW_IND(fn_enum,fn_count) (14ull*(fn_count)+5ull*(fn_enum)+1ull)
  #define F_T_IND(fn_enum,fn_count)  (14ull*(fn_count)+5ull*(fn_enum)+2ull)
  #define F_TE_IND(fn_enum,fn_count) (14ull*(fn_count)+5ull*(fn_enum)+3ull)
  #define F_TN_IND(fn_enum,fn_count) (14ull*(fn_count)+5ull*(fn_enum)+4ull)
#endif
//---------------------------------------------------------------------------
// 5. Bundle C data layout
#ifdef BUNDLE_C_DATA_LAYOUT
  #define F_BS_IND(fn_enum,fn_count) (5ull*(fn_enum))
  #define F_BW_IND(fn_enum,fn_count) (5ull*(fn_enum)+1ull)
  #define F_B_IND(fn_enum,fn_count)  (5ull*(fn_enum)+2ull)
  #define F_BE_IND(fn_enum,fn_count) (5ull*(fn_enum)+3ull)
  #define F_BN_IND(fn_enum,fn_count) (5ull*(fn_enum)+4ull)
  #define F_SW_IND(fn_enum,fn_count) (5ull*(fn_count)+9ull*(fn_enum))
  #define F_S_IND(fn_enum,fn_count)  (5ull*(fn_count)+9ull*(fn_enum)+1ull)
  #define F_SE_IND(fn_enum,fn_count) (5ull*(fn_count)+9ull*(fn_enum)+2ull)
  #define F_W_IND(fn_enum,fn_count)  (5ull*(fn_count)+9ull*(fn_enum)+3ull)
  #define F_C_IND(fn_enum,fn_count)  (5ull*(fn_count)+9ull*(fn_enum)+4ull)
  #define F_E_IND(fn_enum,fn_count)  (5ull*(fn_count)+9ull*(fn_enum)+5ull)
  #define F_NW_IND(fn_enum,fn_count) (5ull*(fn_count)+9ull*(fn_enum)+6ull)
  #define F_N_IND(fn_enum,fn_count)  (5ull*(fn_count)+9ull*(fn_enum)+7ull)
  #define F_NE_IND(fn_enum,fn_count) (5ull*(fn_count)+9ull*(fn_enum)+8ull)
  #define F_TS_IND(fn_enum,fn_count) (14ull*(fn_count)+5ull*(fn_enum))
  #define F_TW_IND(fn_enum,fn_count) (14ull*(fn_count)+5ull*(fn_enum)+1ull)
  #define F_T_IND(fn_enum,fn_count)  (14ull*(fn_count)+5ull*(fn_enum)+2ull)
  #define F_TE_IND(fn_enum,fn_count) (14ull*(fn_count)+5ull*(fn_enum)+3ull)
  #define F_TN_IND(fn_enum,fn_count) (14ull*(fn_count)+5ull*(fn_enum)+4ull)
#endif
//---------------------------------------------------------------------------
namespace D3Q19
{
  const unsigned char BS = 0;
  const unsigned char BW = 1;
  const unsigned char B  = 2;
  const unsigned char BE = 3;
  const unsigned char BN = 4;
  const unsigned char SW = 5;
  const unsigned char S  = 6;
  const unsigned char SE = 7;
  const unsigned char W  = 8;
  const unsigned char C  = 9;
  const unsigned char E  = 10;
  const unsigned char NW = 11;
  const unsigned char N  = 12;
  const unsigned char NE = 13;
  const unsigned char TS = 14;
  const unsigned char TW = 15;
  const unsigned char T  = 16;
  const unsigned char TE = 17;
  const unsigned char TN = 18;
  const unsigned char Q  = 19;

  const unsigned char RDIR[Q] = {TN,TE,T,TW,TS,NE,N,NW,E,C,W,
                                 SE,S,SW,BN,BE,B,BW,BS};

  const char CI[Q][3] = {{0,-1,-1},{-1,0,-1},{0,0,-1},{1,0,-1},{0,1,-1},
                         {-1,-1,0},{0,-1,0},{1,-1,0},{-1,0,0},{0,0,0},
                         {1,0,0},{-1,1,0},{0,1,0},{1,1,0},{0,-1,1},
                         {-1,0,1},{0,0,1},{1,0,1},{0,1,1}};

  const double WEQ0 = 12.0/36.0;
  const double WEQ1 =  2.0/36.0;
  const double WEQ2 =  1.0/36.0;

  const double WI[Q] = {WEQ2,WEQ2,WEQ1,WEQ2,WEQ2, WEQ2,WEQ1,WEQ2,
                        WEQ1,WEQ0,WEQ1, WEQ2,WEQ1,WEQ2,
                        WEQ2,WEQ2,WEQ1,WEQ2,WEQ2};
 
  const double CT = 1.0/sqrt(3.0);
  const double CT2 = CT*CT;
  const double INV_CT2 = 1.0/CT2;
  //-------------------------------------------------------------------------
  inline void get_kin_proj1(double KI1[][3])
  {
    for(unsigned l = 0; l < Q; ++l)
    {
      double dcix = CI[l][X], dciy = CI[l][Y], dciz = CI[l][Z];

      KI1[l][X] = WI[l]*INV_CT2*dcix;
      KI1[l][Y] = WI[l]*INV_CT2*dciy;
      KI1[l][Z] = WI[l]*INV_CT2*dciz;
    }
  }
  //-------------------------------------------------------------------------
  inline void get_kin_proj2(double KI2[][3][3])
  {
    for(unsigned l = 0; l < Q; ++l)
    {
      double dcix = CI[l][X], dciy = CI[l][Y], dciz = CI[l][Z];
      
      KI2[l][X][X] = WI[l]*0.5*INV_CT2*INV_CT2*(dcix*dcix - CT2);
      KI2[l][X][Y] = WI[l]*0.5*INV_CT2*INV_CT2*(dcix*dciy);
      KI2[l][X][Z] = WI[l]*0.5*INV_CT2*INV_CT2*(dcix*dciz);
      
      KI2[l][Y][X] = KI2[l][X][Y];
      KI2[l][Y][Y] = WI[l]*0.5*INV_CT2*INV_CT2*(dciy*dciy - CT2);
      KI2[l][Y][Z] = WI[l]*0.5*INV_CT2*INV_CT2*(dciy*dciz);

      KI2[l][Z][X] = KI2[l][X][Z];
      KI2[l][Z][Y] = KI2[l][Y][Z];
      KI2[l][Z][Z] = WI[l]*0.5*INV_CT2*INV_CT2*(dciz*dciz - CT2);
    }
  }
  //-------------------------------------------------------------------------
  inline double eq1_l(unsigned char l, double den, double ux,
                      double uy, double uz)
  {
    double dcix = CI[l][X], dciy = CI[l][Y], dciz = CI[l][Z];

    return WI[l]*den*(1.0 + INV_CT2*(dcix*ux + dciy*uy + dciz*uz));
  }
  //-------------------------------------------------------------------------
  inline double eq2_l(unsigned char l, double den, double ux,
                      double uy, double uz)
  {
    double dcix = CI[l][X], dciy = CI[l][Y], dciz = CI[l][Z],
      ki2_xx = dcix*dcix - CT2, ki2_xy = dcix*dciy, ki2_xz = dcix*dciz,
      ki2_yy = dciy*dciy - CT2, ki2_yz = dciy*dciz,
      ki2_zz = dciz*dciz - CT2,

      mom1 = INV_CT2*(dcix*ux + dciy*uy + dciz*uz),

      mom2 = 0.5*INV_CT2*INV_CT2*(ki2_xx*ux*ux + ki2_yy*uy*uy + ki2_zz*uz*uz
               + 2.0*(ki2_xy*ux*uy + ki2_xz*ux*uz + ki2_yz*uy*uz));

    return WI[l]*den*(1.0 + mom1 + mom2);
  }
  //-------------------------------------------------------------------------
  inline double frc1_l(unsigned char l, double ux, double uy, double uz,
                       double fx, double fy, double fz)
  {
    double dcix = CI[l][X], dciy = CI[l][Y], dciz = CI[l][Z];

    return WI[l]*INV_CT2*(dcix*fx + dciy*fy + dciz*fz);
  }
  //-------------------------------------------------------------------------
  inline double frc2_l(unsigned char l, double ux, double uy, double uz,
                       double fx, double fy, double fz)
  {
    double dcix = CI[l][X], dciy = CI[l][Y], dciz = CI[l][Z],
      ki2_xx = dcix*dcix - CT2, ki2_xy = dcix*dciy, ki2_xz = dcix*dciz,
      ki2_yy = dciy*dciy - CT2, ki2_yz = dciy*dciz,
      ki2_zz = dciz*dciz - CT2,
      
      m2_xx = 2.0*fx*ux, m2_yy = 2.0*fy*uy, m2_zz = 2.0*fz*uz,
      m2_xy = (fx*uy + fy*ux), m2_xz = (fx*uz + fz*ux),
      m2_yz = (fy*uz + fz*uy),

      mom1 = INV_CT2*(dcix*fx + dciy*fy + dciz*fz),

      mom2 = 0.5*INV_CT2*INV_CT2*(ki2_xx*m2_xx + ki2_yy*m2_yy + ki2_zz*m2_zz
               + 2.0*(ki2_xy*m2_xy + ki2_xz*m2_xz + ki2_yz*m2_yz));

    return WI[l]*(mom1 + mom2);
  }
  //-------------------------------------------------------------------------
  inline void eq1(double den, double ux, double uy, double uz, double *feq)
  {
    double jx = den*ux, jy = den*uy, jz = den*uz,
      cmmn1 = WEQ1*den, cmmn2 = WEQ2*den;

    double cmmn_o_bs = -WEQ2*INV_CT2*(jy + jz);
    feq[BS] = cmmn2 + cmmn_o_bs;
    feq[TN] = cmmn2 - cmmn_o_bs;
  
    double cmmn_o_bw = -WEQ2*INV_CT2*(jx + jz);
    feq[BW] = cmmn2 + cmmn_o_bw;
    feq[TE] = cmmn2 - cmmn_o_bw;
  
    double cmmn_o_b = -WEQ1*INV_CT2*jz;
    feq[B] = cmmn1 + cmmn_o_b;
    feq[T] = cmmn1 - cmmn_o_b;

    double cmmn_o_be = WEQ2*INV_CT2*(jx - jz);
    feq[BE] = cmmn2 + cmmn_o_be;
    feq[TW] = cmmn2 - cmmn_o_be;

    double cmmn_o_bn = WEQ2*INV_CT2*(jy - jz);
    feq[BN] = cmmn2 + cmmn_o_bn;
    feq[TS] = cmmn2 - cmmn_o_bn;
  
    double cmmn_o_sw = -WEQ2*INV_CT2*(jx + jy);
    feq[SW] = cmmn2 + cmmn_o_sw;
    feq[NE] = cmmn2 - cmmn_o_sw;
  
    double cmmn_o_s = -WEQ1*INV_CT2*jy;
    feq[S] = cmmn1 + cmmn_o_s;
    feq[N] = cmmn1 - cmmn_o_s;
  
    double cmmn_o_se = WEQ2*INV_CT2*(jx - jy);
    feq[SE] = cmmn2 + cmmn_o_se;
    feq[NW] = cmmn2 - cmmn_o_se;

    double cmmn_o_w = -WEQ1*INV_CT2*jx;
    feq[W] = cmmn1 + cmmn_o_w;
    feq[E] = cmmn1 - cmmn_o_w;

    feq[C] = WEQ0*den;
  }
  //-------------------------------------------------------------------------
  // Total of 100 floating-point operations (including constant operations)
  //-------------------------------------------------------------------------
  inline void eq2(double den, double ux, double uy, double uz, double *feq)
  {
    const double cff2 = 0.5*INV_CT2*INV_CT2;
  
    double jx = den*ux, jy = den*uy, jz = den*uz,
      m2_xx = cff2*jx*ux, m2_yy = cff2*jy*uy,
      m2_zz = cff2*jz*uz, m2_xy = cff2*jx*uy,
      m2_xz = cff2*jx*uz, m2_yz = cff2*jy*uz,
      cmmn = den - CT2*(m2_xx + m2_yy + m2_zz);

    double cmmn_o_bs = -WEQ2*INV_CT2*(jy + jz),
      cmmn_e_bs = WEQ2*(cmmn + m2_yy + m2_zz + 2.0*m2_yz);

    feq[BS] = cmmn_e_bs + cmmn_o_bs;
    feq[TN] = cmmn_e_bs - cmmn_o_bs;
  
    double cmmn_o_bw = -WEQ2*INV_CT2*(jx + jz),
      cmmn_e_bw = WEQ2*(cmmn + m2_xx + m2_zz + 2.0*m2_xz);

    feq[BW] = cmmn_e_bw + cmmn_o_bw;
    feq[TE] = cmmn_e_bw - cmmn_o_bw;
  
    double cmmn_o_b = -WEQ1*INV_CT2*jz,
      cmmn_e_b = WEQ1*(cmmn + m2_zz);

    feq[B] = cmmn_e_b + cmmn_o_b;
    feq[T] = cmmn_e_b - cmmn_o_b;

    double cmmn_o_be = WEQ2*INV_CT2*(jx - jz),
      cmmn_e_be = WEQ2*(cmmn + m2_xx + m2_zz - 2.0*m2_xz);

    feq[BE] = cmmn_e_be + cmmn_o_be;
    feq[TW] = cmmn_e_be - cmmn_o_be;

    double cmmn_o_bn = WEQ2*INV_CT2*(jy - jz),
      cmmn_e_bn = WEQ2*(cmmn + m2_yy + m2_zz - 2.0*m2_yz);

    feq[BN] = cmmn_e_bn + cmmn_o_bn;
    feq[TS] = cmmn_e_bn - cmmn_o_bn;
  
    double cmmn_o_sw = -WEQ2*INV_CT2*(jx + jy),
      cmmn_e_sw = WEQ2*(cmmn + m2_xx + m2_yy + 2.0*m2_xy);

    feq[SW] = cmmn_e_sw + cmmn_o_sw;
    feq[NE] = cmmn_e_sw - cmmn_o_sw;
  
    double cmmn_o_s = -WEQ1*INV_CT2*jy,
      cmmn_e_s = WEQ1*(cmmn + m2_yy);

    feq[S] = cmmn_e_s + cmmn_o_s;
    feq[N] = cmmn_e_s - cmmn_o_s;
  
    double cmmn_o_se = WEQ2*INV_CT2*(jx - jy),
      cmmn_e_se = WEQ2*(cmmn + m2_xx + m2_yy - 2.0*m2_xy);

    feq[SE] = cmmn_e_se + cmmn_o_se;
    feq[NW] = cmmn_e_se - cmmn_o_se;

    double cmmn_o_w = -WEQ1*INV_CT2*jx,
      cmmn_e_w = WEQ1*(cmmn + m2_xx);

    feq[W] = cmmn_e_w + cmmn_o_w;
    feq[E] = cmmn_e_w - cmmn_o_w;

    feq[C] = WEQ0*cmmn;
  }
  //-------------------------------------------------------------------------
  inline void frc1(double ux, double uy, double uz, double fx, double fy,
                   double fz, double *fext)
  {
    double cmmn_o_bs = -WEQ2*INV_CT2*(fy + fz);
    fext[BS] = cmmn_o_bs;
    fext[TN] = -cmmn_o_bs;
  
    double cmmn_o_bw = -WEQ2*INV_CT2*(fx + fz);
    fext[BW] = cmmn_o_bw;
    fext[TE] = -cmmn_o_bw;
  
    double cmmn_o_b = -WEQ1*INV_CT2*fz;
    fext[B] = cmmn_o_b;
    fext[T] = -cmmn_o_b;

    double cmmn_o_be = WEQ2*INV_CT2*(fx - fz);
    fext[BE] = cmmn_o_be;
    fext[TW] = -cmmn_o_be;

    double cmmn_o_bn = WEQ2*INV_CT2*(fy - fz);
    fext[BN] = cmmn_o_bn;
    fext[TS] = -cmmn_o_bn;
  
    double cmmn_o_sw = -WEQ2*INV_CT2*(fx + fy);
    fext[SW] = cmmn_o_sw;
    fext[NE] = -cmmn_o_sw;
  
    double cmmn_o_s = -WEQ1*INV_CT2*fy;
    fext[S] = cmmn_o_s;
    fext[N] = -cmmn_o_s;
  
    double cmmn_o_se = WEQ2*INV_CT2*(fx - fy);
    fext[SE] = cmmn_o_se;
    fext[NW] = -cmmn_o_se;

    double cmmn_o_w = -WEQ1*INV_CT2*fx;
    fext[W] = cmmn_o_w;
    fext[E] = -cmmn_o_w;

    fext[C] = 0.0;
  }
  //-------------------------------------------------------------------------
  // Total of 109 floating-point operations (including constant operations)
  //-------------------------------------------------------------------------
  inline void frc2(double ux, double uy, double uz, double fx, double fy,
                   double fz, double *fext)
  {
    const double cff2 = 0.5*INV_CT2*INV_CT2;
  
    double m2_xx = cff2*2.0*fx*ux, m2_yy = cff2*2.0*fy*uy,
      m2_zz = cff2*2.0*fz*uz, m2_xy = cff2*(fx*uy + fy*ux),
      m2_xz = cff2*(fx*uz + fz*ux), m2_yz = cff2*(fy*uz + fz*uy),
      cmmn = -CT2*(m2_xx + m2_yy + m2_zz);

    double cmmn_o_bs = -WEQ2*INV_CT2*(fy + fz),
      cmmn_e_bs = WEQ2*(cmmn + m2_yy + m2_zz + 2.0*m2_yz);

    fext[BS] = cmmn_e_bs + cmmn_o_bs;
    fext[TN] = cmmn_e_bs - cmmn_o_bs;
  
    double cmmn_o_bw = -WEQ2*INV_CT2*(fx + fz),
      cmmn_e_bw = WEQ2*(cmmn + m2_xx + m2_zz + 2.0*m2_xz);

    fext[BW] = cmmn_e_bw + cmmn_o_bw;
    fext[TE] = cmmn_e_bw - cmmn_o_bw;
  
    double cmmn_o_b = -WEQ1*INV_CT2*fz,
      cmmn_e_b = WEQ1*(cmmn + m2_zz);

    fext[B] = cmmn_e_b + cmmn_o_b;
    fext[T] = cmmn_e_b - cmmn_o_b;

    double cmmn_o_be = WEQ2*INV_CT2*(fx - fz),
      cmmn_e_be = WEQ2*(cmmn + m2_xx + m2_zz - 2.0*m2_xz);

    fext[BE] = cmmn_e_be + cmmn_o_be;
    fext[TW] = cmmn_e_be - cmmn_o_be;

    double cmmn_o_bn = WEQ2*INV_CT2*(fy - fz),
      cmmn_e_bn = WEQ2*(cmmn + m2_yy + m2_zz - 2.0*m2_yz);

    fext[BN] = cmmn_e_bn + cmmn_o_bn;
    fext[TS] = cmmn_e_bn - cmmn_o_bn;
  
    double cmmn_o_sw = -WEQ2*INV_CT2*(fx + fy),
      cmmn_e_sw = WEQ2*(cmmn + m2_xx + m2_yy + 2.0*m2_xy);

    fext[SW] = cmmn_e_sw + cmmn_o_sw;
    fext[NE] = cmmn_e_sw - cmmn_o_sw;
  
    double cmmn_o_s = -WEQ1*INV_CT2*fy,
      cmmn_e_s = WEQ1*(cmmn + m2_yy);

    fext[S] = cmmn_e_s + cmmn_o_s;
    fext[N] = cmmn_e_s - cmmn_o_s;
  
    double cmmn_o_se = WEQ2*INV_CT2*(fx - fy),
      cmmn_e_se = WEQ2*(cmmn + m2_xx + m2_yy - 2.0*m2_xy);

    fext[SE] = cmmn_e_se + cmmn_o_se;
    fext[NW] = cmmn_e_se - cmmn_o_se;

    double cmmn_o_w = -WEQ1*INV_CT2*fx,
      cmmn_e_w = WEQ1*(cmmn + m2_xx);

    fext[W] = cmmn_e_w + cmmn_o_w;
    fext[E] = cmmn_e_w - cmmn_o_w;

    fext[C] = WEQ0*cmmn;
  }
  //-------------------------------------------------------------------------
  // Total of 37 floating-point operations
  //-------------------------------------------------------------------------
  inline double den_vel_moms(double *fi, double *den, double *ux, double *uy,
    double *uz)
  {
    double neg_z_sum = fi[BS] + fi[BW] + fi[B] + fi[BE] + fi[BN],
      pos_z_sum = fi[TS] + fi[TW] + fi[T] + fi[TE] + fi[TN],
      neg_y_part_sum = fi[SW] + fi[S] + fi[SE],
      pos_y_part_sum = fi[NW] + fi[N] + fi[NE],
      
      tmp_den = neg_z_sum + neg_y_part_sum + pos_y_part_sum + pos_z_sum
                + fi[W] + fi[C] + fi[E],
      inv_den = 1.0/tmp_den; 

    (*den) = tmp_den;
    
    (*ux) = inv_den*((fi[BE] + fi[SE] + fi[E] + fi[NE] + fi[TE])
                - (fi[BW] + fi[SW] + fi[W] + fi[NW] + fi[TW]));
      
    (*uy) = inv_den*((pos_y_part_sum + fi[BN] + fi[TN])
                - (neg_y_part_sum + fi[BS] + fi[TS]));

    (*uz) = inv_den*(pos_z_sum - neg_z_sum);
      
    return inv_den;
  }
}
//===========================================================================
#endif
//===========================================================================