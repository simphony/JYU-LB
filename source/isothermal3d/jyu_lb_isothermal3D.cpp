//===========================================================================
// Author: Keijo Mattila, JYU, February 2015
// Description: implementation of LBM solvers
// Details: 
//===========================================================================
#include <omp.h>
#include <math.h>
#include <cstring>
#include <iostream>
#include <sys/time.h>
#include "lb_const.h"
#include "lb_raw_io.h"
#include "lb_input_parser.h"
//===========================================================================
typedef LB_Input_Parser::Isothermal3D Parser;
typedef LB_Solver::Isothermal3D::FlowInfo FlowInfo;
typedef LB_Solver::Isothermal3D::FlowParams FlowParams;
typedef LB_Solver::Isothermal3D::TimeStepper TimeStepper;
//===========================================================================
using namespace std;
//===========================================================================
void open_evol_file(Parser *par, ofstream &evol_file)
{
  string fname(par->base_io_fname());
  fname += ".evol";
  evol_file.open(fname.c_str());

  if( !evol_file.is_open() ) {
    printf("Warning: can't open file %s for evol.output\n!", fname.c_str());
  } else {
    evol_file << "#1. Tstep \t 2. t \t 3. Tot.den \t";
    evol_file << "4. Tot.jx \t 5. Tot.jy \t 6. Tot.jz \t 7. Tot.j2 \t";
    evol_file << "8. Ave.ux \t 9. Ave.uy \t 10. Ave.uz \t 11. Ave.u2 \t";
    evol_file << "12. Max.u2 \t 13. Max.den \t 14. Min.den" << endl;
    evol_file.precision(12);
    evol_file.setf(ios::scientific);
  }
}
//---------------------------------------------------------------------------
void output_evol_info(ofstream &evol_file, double dtstep, unsigned int tstep,
                      FlowInfo *finfo)
{
  evol_file << tstep << "\t";
  evol_file << dtstep << "\t";
  evol_file << finfo->tot_den << "\t";
  evol_file << finfo->tot_jx << "\t";
  evol_file << finfo->tot_jy << "\t";
  evol_file << finfo->tot_jz << "\t";
  evol_file << finfo->tot_j2 << "\t";
  evol_file << finfo->ave_ux << "\t";
  evol_file << finfo->ave_uy << "\t";
  evol_file << finfo->ave_uz << "\t";
  evol_file << finfo->ave_u2 << "\t";
  evol_file << finfo->max_u2 << "\t";
  evol_file << finfo->max_den << "\t";
  evol_file << finfo->min_den << endl;
}
//---------------------------------------------------------------------------
void print_input_params(Parser *par, FlowParams *fprms)
{
  printf("\n");
  printf("==============================================================\n");
  printf("JYU-LB Isothermal 3D Fluid Flow Simulator\n");
  printf("==============================================================\n");
  printf("Base I/O file name = %s\n", par->base_io_fname());
  printf("nx = %d, ny = %d, nz = %d\n", par->nx(), par->ny(), par->nz());
  printf("dr = %f, dt = %f\n", fprms->dr, fprms->dt);
  printf("ref.den = %f, kvisc. = %f\n", fprms->ref_den, fprms->kvisc);
  printf("gx = %f, gy = %f, gz = %f\n", fprms->gx, fprms->gy, fprms->gz);
  printf("External forcing = %d\n", fprms->external_forcing);
  printf("Flow type = %d\n", fprms->flow_type);
  printf("Collision operator = %d\n", fprms->collision_operator);
  printf("Number of time steps = %d\n", par->tsteps());
  printf("Evolution info period = %d\n", par->evol_period());
  printf("Execution info = %s\n", par->exec_info());
  printf("==============================================================\n");
  int nthreads;
  #pragma omp parallel shared(nthreads)
  {
    #pragma omp master
    {
      nthreads = omp_get_num_threads();
    }
  }
  printf("Using %d OpenMP threads for computation\n",nthreads);
  printf("==============================================================\n");
}
//---------------------------------------------------------------------------
void set_uniform_init_den(FlowParams *fprms, Lattice::Isothermal3D *lat)
{
  double ref_den = fprms->ref_den, den0 = ref_den*1.0;
  
  #pragma omp parallel for
  for(unsigned int k = 0; k < lat->nz(); ++k) {
    for(unsigned int j = 0; j < lat->ny(); ++j) {
      for(unsigned int i = 1; i < (lat->nx()-1); ++i)
      {
        lat->set_den_ijk(i,j,k,ref_den);
      }
    }
  }
}
//---------------------------------------------------------------------------
void set_uniform_init_vel(FlowParams *fprms, Lattice::Isothermal3D *lat)
{
  double cr = fprms->dr/fprms->dt,
    ux0 = cr*0.0, uy0 = cr*0.0, uz0 = cr*0.0;
  
  #pragma omp parallel for
  for(unsigned int k = 0; k < lat->nz(); ++k) {
    for(unsigned int j = 0; j < lat->ny(); ++j) {
      for(unsigned int i = 1; i < (lat->nx()-1); ++i)
      {
        lat->set_vel_ijk(i,j,k,ux0,uy0,uz0);
      }
    }
  }
}
//===========================================================================
// MAIN FUNCTION
//===========================================================================
int main(int argc, char *argv[])
{
  FlowParams fparams;
  Parser par(argc, argv, &fparams);
  
  ofstream evol_file;
  open_evol_file(&par, evol_file);
  print_input_params(&par, &fparams);
    
  Lattice::Isothermal3D lat(par.nx(),par.ny(),par.nz());
  LB_Raw_IO::read_geom(par.base_io_fname(), &lat);

  if( !LB_Raw_IO::read_den(par.base_io_fname(), &lat) )
    set_uniform_init_den(&fparams, &lat);

  if( !LB_Raw_IO::read_vel(par.base_io_fname(), &lat) )
    set_uniform_init_vel(&fparams, &lat);
  
  if( fparams.external_forcing )
    LB_Raw_IO::read_frc(par.base_io_fname(), &lat);

  FlowInfo finfo;
  TimeStepper solver(&lat, &fparams);  

  unsigned int eperiod = par.evol_period(), tsteps = par.tsteps(),
    eperiod_count = tsteps/eperiod, tsteps_rem = tsteps%eperiod;
    
  solver.calc_flow_info(&lat, &finfo);
  printf("tstep %d, tot.mass = %e, max.u2 = %e\n",
         0, finfo.tot_den, finfo.max_u2);
  output_evol_info(evol_file, 0, 0.0, &finfo);

  struct timeval start, end;
  long comp_time_milliseconds, seconds, microseconds;    
  gettimeofday(&start, NULL);

  for(unsigned int ep = 1; ep <= eperiod_count; ++ep) {
    solver.evolve(&lat,eperiod);
    solver.calc_flow_info(&lat, &finfo);
    
    unsigned int tstep = ep*eperiod;
    printf("tstep %d, tot.mass = %e, max.u2 = %e\n",
           tstep, finfo.tot_den,finfo.max_u2);
    output_evol_info(evol_file, tstep, fparams.dt*tstep, &finfo);
  }
  if( tsteps_rem > 0 ) {
    solver.evolve(&lat,tsteps_rem);
    solver.calc_flow_info(&lat, &finfo);
    printf("tstep %d, tot.mass = %e, max.u2 = %e\n",
           tsteps, finfo.tot_den,finfo.max_u2);
    output_evol_info(evol_file, tsteps, fparams.dt*tsteps, &finfo);
  }
  gettimeofday(&end, NULL);
  seconds  = end.tv_sec  - start.tv_sec;
  microseconds = end.tv_usec - start.tv_usec;
  comp_time_milliseconds = ((seconds) * 1000 + microseconds/1000.0) + 0.5;

  double dnx = lat.nx(), dny = lat.ny(), dnz = lat.nz(), dts = tsteps,
    KFLUP = (dnx*dny*dnz*dts)/(double)1e3;
  
  printf("==============================================================\n");
  printf("Computational time: %ld milliseconds\n", comp_time_milliseconds);
  printf("--------------------------------------------------------------\n");
  printf("MFLUPS: %f\n", KFLUP/(double)comp_time_milliseconds);
  printf("==============================================================\n");
  
  LB_Raw_IO::write_den(par.base_io_fname(), &lat);
  LB_Raw_IO::write_vel(par.base_io_fname(), &lat);

  evol_file.close();
  return 0;
}
//===========================================================================