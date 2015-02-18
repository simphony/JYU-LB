//===========================================================================
// Author: Keijo Mattila, JYU, February 2015
// Description: Definition of LBM constants
// Details: Prescribes the data layout used for distribution functions
//===========================================================================
#ifndef LB_CONST_H
#define LB_CONST_H
//===========================================================================
//#define COLL_OPT_DATA_LAYOUT
//#define PROP_OPT_DATA_LAYOUT
//#define BUNDLE_A_DATA_LAYOUT
//#define BUNDLE_B_DATA_LAYOUT
#define BUNDLE_C_DATA_LAYOUT
//===========================================================================
namespace LB_Const
{
  const unsigned char X = 0;
  const unsigned char Y = 1;
  const unsigned char Z = 2;

  const unsigned char SOLID_NODE = 0;
  const unsigned char FLUID_NODE = 255;

  const unsigned char STOKES_FLOW = 0;
  const unsigned char LAMINAR_FLOW = 1;
  const unsigned char TURBULENT_FLOW = 2;

  const unsigned char BGK = 0;
  const unsigned char TRT = 1;
  const unsigned char MRT = 2;
  const unsigned char REG = 3;
}
//===========================================================================
#endif
//===========================================================================