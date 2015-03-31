//===========================================================================
// Author: Keijo Mattila, JYU, February 2015
// Description: Input script parser for LBM flow simulators
// Details: 
//===========================================================================
#include <limits>
#include "const.h"
#include "input_parser.h"
//===========================================================================
using namespace std;
//===========================================================================
IsothermalInputParser::IsothermalInputParser(int argc, char *argv[],
  IsothermalFlowParams *params)
{
  // Set default values
  params->dr = 1.0;
  params->dt = 1.0;
  params->ref_den = 1.0;
  params->kvisc = 1.0/6.0;
  params->gx = 0.0;
  params->gy = 0.0;
  params->gz = 0.0;
  params->flow_type = STOKES_FLOW;
  params->collision_operator = BGK;
  params->external_forcing = false;
  
  a_nx = 1; a_ny = 1; a_nz = 1;
  a_evol_info_period = 100;
  a_time_steps = 1000;
  
  a_base_io_fname = "jyu_lb_isothermal";
  a_exec_info = "";

  if( argc < 2 ) {
    printf("Warning: not enough input arguments!\n");
    printf("Warning: using default simulation parameters!\n");
    return;
  }
  parse_input_script(argc, argv, params);
}
//---------------------------------------------------------------------------
IsothermalInputParser::~IsothermalInputParser()
{
}
//---------------------------------------------------------------------------
char IsothermalInputParser::ignore_comments(ifstream &ifile) const
{
  char input_chr = ifile.peek();
  while( input_chr == '#' ) {
    ifile.ignore(numeric_limits<streamsize>::max(), '\n');
    input_chr = ifile.peek();
  }
  return input_chr;
}
//---------------------------------------------------------------------------
void IsothermalInputParser::parse_input_script(int argc, char *argv[],
  IsothermalFlowParams *params)
{
  ifstream ifile(argv[1]);
  if( !ifile.is_open() ) {
    printf("Warning: unable to open input script!");
    printf("Warning: using default simulation parameters!\n");
    return;
  }
  char input_chr;
  double input_double;
  string input_str;
  int input_int;
  unsigned int input_uint;

  // read base name of the I/O data files
  ignore_comments(ifile);
  getline(ifile, input_str);
  set_base_io_fname(input_str.c_str());
  ifile.ignore(numeric_limits<streamsize>::max(), '\n');

  // read lattice size
  ignore_comments(ifile);
  ifile >> input_uint; set_nx(input_uint);
  ifile >> input_uint; set_ny(input_uint);
  ifile >> input_uint; set_nz(input_uint);
  ifile.ignore(numeric_limits<streamsize>::max(), '\n');
  
  // read lattice spacing
  ignore_comments(ifile);
  ifile >> input_double; set_lattice_spacing(input_double,params);
  ifile.ignore(numeric_limits<streamsize>::max(), '\n');
  
  // read discrete time step
  ignore_comments(ifile);
  ifile >> input_double; set_discrete_time_step(input_double,params);
  ifile.ignore(numeric_limits<streamsize>::max(), '\n');

  // read reference density
  ignore_comments(ifile);
  ifile >> input_double; set_ref_den(input_double,params);
  ifile.ignore(numeric_limits<streamsize>::max(), '\n');
  
  // read kinematic viscosity
  ignore_comments(ifile);
  ifile >> input_double; set_kvisc(input_double,params);
  ifile.ignore(numeric_limits<streamsize>::max(), '\n');

  // read gravity
  ignore_comments(ifile);
  ifile >> input_double; params->gx = input_double;
  ifile >> input_double; params->gy = input_double;
  ifile >> input_double; params->gz = input_double;
  ifile.ignore(numeric_limits<streamsize>::max(), '\n');
  
  // read additional external forcing option
  ignore_comments(ifile);
  ifile >> input_uint; set_extern_forcing(input_uint,params);
  ifile.ignore(numeric_limits<streamsize>::max(), '\n');

  // read flow type
  ignore_comments(ifile);
  ifile >> input_uint; set_flow_type(input_uint,params);
  ifile.ignore(numeric_limits<streamsize>::max(), '\n');

  // read collision operator
  ignore_comments(ifile);
  ifile >> input_uint; set_collision_operator(input_uint,params);
  ifile.ignore(numeric_limits<streamsize>::max(), '\n');
  
  // read number of discrete time steps
  ignore_comments(ifile);
  ifile >> input_uint; set_tstep_count(input_uint);
  ifile.ignore(numeric_limits<streamsize>::max(), '\n');

  // read evolution info output interval
  ignore_comments(ifile);
  ifile >> input_uint; set_evol_info_interval(input_uint);
  ifile.ignore(numeric_limits<streamsize>::max(), '\n');
  
  // read execution info
  ignore_comments(ifile);
  getline(ifile, input_str);
  set_exec_info(input_str.c_str());
}
//---------------------------------------------------------------------------
void IsothermalInputParser::set_base_io_fname(const char *fname)
{
  a_base_io_fname = fname;

  a_base_io_fname.erase(0, a_base_io_fname.find_first_not_of(" \t\v\n\r"));
  int first = a_base_io_fname.find_first_of(" \t\v\n\r");
  if( first != string::npos ) {
    int count = a_base_io_fname.size() - first;
    a_base_io_fname.erase(first, count);
  }
}
//---------------------------------------------------------------------------
void IsothermalInputParser::set_nx(unsigned int size)
{
  if( size < 1 ) {
    printf("Warning: invalid lattice size argument!");
    printf("Warning: using default lattice size!\n");
    return;
  }
  a_nx = size;
}
//---------------------------------------------------------------------------
void IsothermalInputParser::set_ny(unsigned int size)
{
  if( size < 1 ) {
    printf("Warning: invalid lattice size argument!");
    printf("Warning: using default lattice size!\n");
    return;
  }
  a_ny = size;
}
//---------------------------------------------------------------------------
void IsothermalInputParser::set_nz(unsigned int size)
{
  if( size < 1 ) {
    printf("Warning: invalid lattice size argument!");
    printf("Warning: using default lattice size!\n");
    return;
  }
  a_nz = size;
}
//---------------------------------------------------------------------------
void IsothermalInputParser::set_lattice_spacing(double dr,
  IsothermalFlowParams *params)
{
  if( dr <= 0.0 ) {
    printf("Warning: lattice spacing must be positive!");
    printf("Warning: using default lattice spacing!\n");
    return;
  }
  params->dr = dr;
}
//---------------------------------------------------------------------------
void IsothermalInputParser::set_discrete_time_step(double dt,
  IsothermalFlowParams *params)
{
  if( dt <= 0.0 ) {
    printf("Warning: discrete time step must be positive!");
    printf("Warning: using default discrete time step!\n");
    return;
  }
  params->dt = dt;
}
//---------------------------------------------------------------------------
void IsothermalInputParser::set_ref_den(double rden,
  IsothermalFlowParams *params)
{
  if( rden <= 0.0 ) {
    printf("Warning: reference density must be positive!");
    printf("Warning: using default reference density!\n");
    return;
  }
  params->ref_den = rden;
}
//---------------------------------------------------------------------------
void IsothermalInputParser::set_kvisc(double kvisc,
  IsothermalFlowParams *params)
{
  if( kvisc <= 0.0 ) {
    printf("Warning: kinematic viscosity must be positive!");
    printf("Warning: using default kinematic viscosity!\n");
    return;
  }
  params->kvisc = kvisc;
}
//---------------------------------------------------------------------------
void IsothermalInputParser::set_extern_forcing(
  unsigned int is_extern_forcing, IsothermalFlowParams *params)
{
  if( is_extern_forcing != 0 && is_extern_forcing != 1 ) {
    printf("Warning: invalid external forcing option argument!");
    printf("Warning: using default external forcing option!\n");
    return;
  }
  params->external_forcing = (is_extern_forcing == 1);
}
//---------------------------------------------------------------------------
void IsothermalInputParser::set_flow_type(unsigned int flow_type,
  IsothermalFlowParams *params)
{
  if( flow_type != STOKES_FLOW &&
      flow_type != LAMINAR_FLOW &&
      flow_type != TURBULENT_FLOW ) {
    printf("Warning: invalid flow type argument!");
    printf("Warning: using default flow type!\n");
    return;
  }
  params->flow_type = flow_type;
}
//---------------------------------------------------------------------------
void IsothermalInputParser::set_collision_operator(unsigned int coll_oper,
  IsothermalFlowParams *params)
{
  if( coll_oper != BGK && coll_oper != TRT &&
      coll_oper != MRT && coll_oper != REG ) {
    printf("Warning: invalid collision operator argument!");
    printf("Warning: using default collision operator!\n");
    return;
  }
  params->collision_operator = coll_oper;
}
//---------------------------------------------------------------------------
void IsothermalInputParser::set_tstep_count(unsigned int tstep_count)
{
  a_time_steps = tstep_count;
}
//---------------------------------------------------------------------------
void IsothermalInputParser::set_evol_info_interval(
  unsigned int evol_info_period)
{
  a_evol_info_period = evol_info_period;
}
//---------------------------------------------------------------------------
void IsothermalInputParser::set_exec_info(const char *exec_info)
{
  a_exec_info = exec_info;

  a_exec_info.erase(0, a_exec_info.find_first_not_of("\t\v\n\r"));
  int first = a_exec_info.find_first_of("\t\v\n\r");
  if( first != string::npos ) {
    int count = a_exec_info.size() - first;
    a_exec_info.erase(first, count);
  }
}
//===========================================================================