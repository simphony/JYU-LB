//===========================================================================
// Author: Keijo Mattila, JYU, March 2015
// Description: 
// Details: 
//===========================================================================
#ifndef DEF_H
#define DEF_H
//===========================================================================
typedef unsigned int UINT;
typedef unsigned long long int ULLINT;
//---------------------------------------------------------------------------
typedef void (*EQ3D_FPTR)(double,double,double,double,double *);
typedef void (*FRC3D_FPTR)(double,double,double,
                           double,double,double,double *);
//---------------------------------------------------------------------------
//#define COLL_OPT_DATA_LAYOUT
//#define PROP_OPT_DATA_LAYOUT
//#define BUNDLE_A_DATA_LAYOUT
//#define BUNDLE_B_DATA_LAYOUT
#define BUNDLE_C_DATA_LAYOUT
//---------------------------------------------------------------------------
#define SAFE_MODE // Enable index checking
//===========================================================================
#endif
//===========================================================================