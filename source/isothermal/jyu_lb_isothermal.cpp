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
#include "node.h"
#include "raw_io.h"
#include "input_parser.h"
//===========================================================================
typedef IsothermalInputParser Parser;
typedef IsothermalFlowInfo FlowInfo;
typedef IsothermalFlowParams FlowParams;
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
    
  double origin[3] = {0,0,0};
  UINT size[3] = {par.nx(),par.ny(),par.nz()};
  Lattice lat(size, origin);
  Geometry geom(&lat);

  IsothermalRawIO raw_io(&lat);
  raw_io.read_geom(par.base_io_fname(), geom.get_phase());

  FlowInfo finfo;
  IsothermalSolver solver(&geom, &fparams);  
  IsothermalNodeData *fdata = solver.get_fluid_node_data();
    
  if( !raw_io.read_den(par.base_io_fname(), fdata) ) {
    printf("No input file for density: using default initial density.\n");
    printf("--------------------------------------------------------------\n");
  }
  if( !raw_io.read_vel(par.base_io_fname(), fdata) ) {
    printf("No input file for velocity: using default initial velocity.\n");
    printf("--------------------------------------------------------------\n");
  }
  if( fparams.external_forcing )
    raw_io.read_frc(par.base_io_fname(), fdata);

  solver.init_field_data();

  unsigned int eperiod = par.evol_period(), tsteps = par.tsteps(),
    eperiod_count = tsteps/eperiod, tsteps_rem = tsteps%eperiod;
    
  solver.calc_flow_info(&finfo);
  printf("tstep %d, tot.mass = %e, max.u2 = %e\n",
         0, finfo.tot_den, finfo.max_u2);
  output_evol_info(evol_file, 0, 0.0, &finfo);

  struct timeval start, end;
  long comp_time_milliseconds, seconds, microseconds;    
  gettimeofday(&start, NULL);

  for(unsigned int ep = 1; ep <= eperiod_count; ++ep) {
    solver.evolve(eperiod);
    solver.calc_flow_info(&finfo);
    
    unsigned int tstep = ep*eperiod;
    printf("tstep %d, tot.mass = %e, max.u2 = %e\n",
           tstep, finfo.tot_den,finfo.max_u2);
    output_evol_info(evol_file, tstep, fparams.dt*tstep, &finfo);
  }
  if( tsteps_rem > 0 ) {
    solver.evolve(tsteps_rem);
    solver.calc_flow_info(&finfo);
    printf("tstep %d, tot.mass = %e, max.u2 = %e\n",
           tsteps, finfo.tot_den,finfo.max_u2);
    output_evol_info(evol_file, tsteps, fparams.dt*tsteps, &finfo);
  }
  gettimeofday(&end, NULL);
  seconds  = end.tv_sec  - start.tv_sec;
  microseconds = end.tv_usec - start.tv_usec;
  comp_time_milliseconds = ((seconds) * 1000 + microseconds/1000.0) + 0.5;

  double dncount = fdata->get_nodeset()->get_node_count(),
    dts = tsteps, KFLUP = (dncount*dts)/(double)1e3;
  
  printf("==============================================================\n");
  printf("Computational time: %ld milliseconds\n", comp_time_milliseconds);
  printf("--------------------------------------------------------------\n");
  printf("MFLUPS: %f\n", KFLUP/(double)comp_time_milliseconds);
  printf("==============================================================\n");
  
  raw_io.write_den(par.base_io_fname(), fdata);
  raw_io.write_vel(par.base_io_fname(), fdata);

  evol_file.close();
  return 0;
}
//===========================================================================