//===========================================================================
// Author: Keijo Mattila, JYU, February 2015
// Description: Input script parser for LBM flow simulators
// Details: 
//===========================================================================
#ifndef INPUT_PARSER_H
#define INPUT_PARSER_H
//===========================================================================
#include <fstream>
#include "solver.h"
//===========================================================================
class IsothermalInputParser
{
  public:
    explicit IsothermalInputParser(int argc, char *argv[],
      IsothermalFlowParams *params);
    ~IsothermalInputParser();
    
    const char *base_io_fname() const {return a_base_io_fname.c_str();}
    unsigned int nx() const {return a_nx;}
    unsigned int ny() const {return a_ny;}
    unsigned int nz() const {return a_nz;}
    unsigned int tsteps() const {return a_time_steps;}
    unsigned int evol_period() const {return a_evol_info_period;}
    const char *exec_info() const {return a_exec_info.c_str();}

  private:
    char ignore_comments(std::ifstream &ifile) const;

    void parse_input_script(int argc, char *argv[],
      IsothermalFlowParams *params);

    unsigned int a_nx, a_ny, a_nz, a_evol_info_period, a_time_steps;
    std::string a_base_io_fname, a_exec_info;

    void set_base_io_fname(const char *fname);
    void set_nx(unsigned int size);
    void set_ny(unsigned int size);
    void set_nz(unsigned int size);

    void set_lattice_spacing(double dr,
      IsothermalFlowParams *params);

    void set_discrete_time_step(double dt,
      IsothermalFlowParams *params);
        
    void set_ref_den(double rden,
      IsothermalFlowParams *params);

    void set_kvisc(double kvisc,
      IsothermalFlowParams *params);
        
    void set_extern_forcing(unsigned int is_extern_forcing,
      IsothermalFlowParams *params);
        
    void set_flow_type(unsigned int flow_type,
      IsothermalFlowParams *params);
        
    void set_collision_operator(unsigned int coll_oper,
      IsothermalFlowParams *params);
        
    void set_tstep_count(unsigned int tstep_count);

    void set_evol_info_interval(unsigned int evol_info_period);
      
    void set_exec_info(const char *info);
};
//===========================================================================
#endif
//===========================================================================
