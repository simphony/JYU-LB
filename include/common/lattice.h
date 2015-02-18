//===========================================================================
// Author: Keijo Mattila, JYU, February 2015
// Description: Implementation of Lattice data structures
// Details: Lattices store geometries + hydrodynamic variables,
//          but not distributions values
//===========================================================================
#ifndef LB_LATTICE_H
#define LB_LATTICE_H
//===========================================================================
namespace Lattice
{
  typedef unsigned int UINT;

  //-------------------------------------------------------------------------
  // Class SparseGeom3D
  //-------------------------------------------------------------------------
  class SparseGeom3D
  {
    public:
      SparseGeom3D(UINT nx, UINT ny, UINT nz);
      virtual ~SparseGeom3D();

      //---------------------------------------------------------------------
      // Set methods
      //---------------------------------------------------------------------
      void set_geom(unsigned char *geom);

      //---------------------------------------------------------------------
      // Get methods
      //---------------------------------------------------------------------
      inline unsigned char get_geom_n(UINT n) const {
        return a_geom[n];
      }
      inline unsigned char get_geom_ijk(UINT i, UINT j, UINT k) const {
        return get_geom_n(IJK_TO_N(i,j,k));
      }
      inline bool get_fnode_enum_n(UINT n, UINT *fnode_enum) const {
        UINT dn = a_fnode_enum[n];
        if( dn == 0 ) return false;

        (*fnode_enum) = (dn - 1);
        return true;
      }
      inline bool get_fnode_enum_ijk(UINT i, UINT j, UINT k,
                                    UINT *fnode_enum) const {
        return get_fnode_enum_n(IJK_TO_N(i,j,k), fnode_enum);
      }
      inline void get_fnode_lat_coord(UINT fnode_enum, UINT *i,
                                      UINT *j, UINT *k) const {
        UINT dn = fnode_enum + 1;
        (*i) = a_li[dn]; (*j) = a_lj[dn]; (*k) = a_lk[dn];
      }
      inline UINT nx() const {return a_nx;}
      inline UINT ny() const {return a_ny;}
      inline UINT nz() const {return a_nz;}
      inline UINT fluid_ncount() const {return a_fluid_ncount;}
      //---------------------------------------------------------------------
      // Functions
      //---------------------------------------------------------------------
      inline UINT IJK_TO_N(UINT i, UINT j, UINT k) const {
        return k*a_nx*a_ny + j*a_nx + i;
      }

    protected:
      UINT a_nx, a_ny, a_nz, a_fluid_ncount;
      unsigned short *a_li, *a_lj, *a_lk;
      unsigned char *a_geom;
      UINT *a_fnode_enum;

      void alloc_geom(UINT nx, UINT ny, UINT nz);
      void alloc_lat_coords(UINT fluid_ncount);
      virtual void alloc_fluid_data() = 0;

      void dealloc_geom();
      void dealloc_lat_coords();
      virtual void dealloc_fluid_data() = 0;
  };
  
  //-------------------------------------------------------------------------
  // Class Isothermal3D
  //-------------------------------------------------------------------------
  class Isothermal3D: public SparseGeom3D
  {
    public:
      Isothermal3D(UINT nx, UINT ny, UINT nz);
      virtual ~Isothermal3D();

      //---------------------------------------------------------------------
      // Set methods
      //---------------------------------------------------------------------
      inline void set_den_n(UINT n, double den) {
        a_den[a_fnode_enum[n]] = den;
      }
      inline void set_den_ijk(UINT i, UINT j, UINT k, double den) {
        set_den_n(IJK_TO_N(i,j,k),den);
      }
      inline void set_vel_n(UINT n, double ux, double uy, double uz) {
        UINT dn = a_fnode_enum[n];
        a_ux[dn] = ux; a_uy[dn] = uy; a_uz[dn] = uz;
      }
      inline void set_vel_ijk(UINT i, UINT j, UINT k, double ux,
                              double uy, double uz) {
        set_vel_n(IJK_TO_N(i,j,k),ux,uy,uz);
      }
      inline void set_force_n(UINT n, double fx, double fy, double fz) {
        UINT dn = a_fnode_enum[n];
        a_fx[dn] = fx; a_fy[dn] = fy; a_fz[dn] = fz;
      }
      inline void set_force_ijk(UINT i, UINT j, UINT k, double fx,
                                double fy, double fz) {
        set_force_n(IJK_TO_N(i,j,k),fx,fy,fz);
      }
      inline void set_fnode_den(UINT fnode_enum, double den) {
        a_den[fnode_enum + 1] = den;
      }
      inline void set_fnode_vel(UINT fnode_enum, double ux,
                                double uy, double uz) {
          UINT dn = fnode_enum + 1;
          a_ux[dn] = ux; a_uy[dn] = uy; a_uz[dn] = uz;
      }
      inline void set_fnode_force(UINT fnode_enum, double fx,
                                  double fy, double fz) {
        UINT dn = fnode_enum + 1;
        a_fx[dn] = fx; a_fy[dn] = fy; a_fz[dn] = fz;
      }
      //---------------------------------------------------------------------
      // Get methods
      //---------------------------------------------------------------------
      inline void get_den_n(UINT n, double *den) const {
        (*den) = a_den[a_fnode_enum[n]];
      }
      inline void get_den_ijk(UINT i, UINT j, UINT k, double *den) const {
        get_den_n(IJK_TO_N(i,j,k),den);
      }
      inline void get_vel_n(UINT n, double *ux, double *uy,
                            double *uz) const {
        UINT dn = a_fnode_enum[n];
        (*ux) = a_ux[dn]; (*uy) = a_uy[dn]; (*uz) = a_uz[dn];
      }
      inline void get_vel_ijk(UINT i, UINT j, UINT k, double *ux,
                              double *uy, double *uz) const {
        get_vel_n(IJK_TO_N(i,j,k),ux,uy,uz);
      }
      inline void get_force_n(UINT n, double *fx, double *fy,
                              double *fz) const {
        UINT dn = a_fnode_enum[n];
        (*fx) = a_fx[dn]; (*fy) = a_fy[dn]; (*fz) = a_fz[dn];
      }
      inline void get_force_ijk(UINT i, UINT j, UINT k, double *fx,
                                double *fy, double *fz) const {
        get_force_n(IJK_TO_N(i,j,k),fx,fy,fz);
      }
      inline void get_fnode_den(UINT fnode_enum, double *den) const {
        (*den) = a_den[fnode_enum + 1];
      }
      inline void get_fnode_vel(UINT fnode_enum, double *ux,
                                double *uy, double *uz) const {
        UINT dn = fnode_enum + 1;
        (*ux) = a_ux[dn]; (*uy) = a_uy[dn]; (*uz) = a_uz[dn];
      }
      inline void get_fnode_force(UINT fnode_enum, double *fx,
                                  double *fy, double *fz) const {
        unsigned int dn = fnode_enum + 1;
        (*fx) = a_fx[dn]; (*fy) = a_fy[dn]; (*fz) = a_fz[dn];
      }

    private:
      double *a_den, *a_ux, *a_uy, *a_uz, *a_fx, *a_fy, *a_fz;

      virtual void alloc_fluid_data();
      virtual void dealloc_fluid_data();
  };
}
//===========================================================================
#endif
//===========================================================================