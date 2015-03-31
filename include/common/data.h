//===========================================================================
// Author: Keijo Mattila, JYU, March 2015
// Description: 
// Details: 
//===========================================================================
#ifndef DATA_H
#define DATA_H
//===========================================================================
#include "node.h"
#include "const.h"
//===========================================================================
typedef NodeData<unsigned char> GeomData;
typedef NodeData<double> FieldData;

typedef DataValueFilter<unsigned char> GeomDataFilter;
typedef DataValueFilter<double> FieldDataFilter;
//===========================================================================
// Class NodeData
//---------------------------------------------------------------------------
template <class T>
class NodeData
{
  public:
    NodeData(NodeSet *nodes, T ivalue)
    {
      _nodes = nodes;
      _ncount = nodes->get_node_count();
      _data = new T[_ncount];
      _ivalue = ivalue;
      for(UINT n = 0; n < _ncount; ++n) {_data[n] = ivalue;}
    }
    ~NodeData() {if( _data != 0 ) delete [] _data;}

    void set_val_n(UINT n, T value) throw(std::out_of_range)
    {
      #ifdef SAFE_MODE
      if( n >= _ncount ) {
        throw std::out_of_range("Enumeration number out of range!");
      }
      #endif
      _data[n] = value;
    }
    void set_val_ijk(const UINT ijk[3], T value) throw(std::out_of_range)
    {
      UINT n = _nodes->get_n(ijk);
      _data[n] = value;
    }
    T get_val_n(UINT n) const throw(std::out_of_range)
    {
      #ifdef SAFE_MODE
      if( n >= _ncount ) {
        throw std::out_of_range("Enumeration number out of range!");
      }
      #endif
      return _data[n];
    }
    T get_val_ijk(const UINT ijk[3]) const throw(std::out_of_range)
    {
      UINT n = _nodes->get_n(ijk);
      return _data[n];
    }
    NodeSet *get_nodeset() const {return _nodes;}

  private:
    NodeSet *_nodes;
    UINT _ncount;
    T _ivalue, *_data;
};
//---------------------------------------------------------------------------
// Class IsothermalNodeData
//---------------------------------------------------------------------------
class IsothermalNodeData
{
  public:
    IsothermalNodeData(NodeSet *nodes, double iden, double ivel[3],
                       double ifrc[3])
    {
      _nodes = nodes;
      _den = new FieldData(_nodes, iden);
      _velx = new FieldData(_nodes, ivel[0]);
      _vely = new FieldData(_nodes, ivel[1]);
      _velz = new FieldData(_nodes, ivel[2]);
      _frcx = new FieldData(_nodes, ifrc[0]);
      _frcy = new FieldData(_nodes, ifrc[1]);
      _frcz = new FieldData(_nodes, ifrc[2]);
      _ivel[0] = ivel[0]; _ivel[1] = ivel[1]; _ivel[2] = ivel[2];
      _ifrc[0] = ifrc[0]; _ifrc[1] = ifrc[1]; _ifrc[2] = ifrc[2];
    }
    ~IsothermalNodeData() {
      if( _den != 0 ) delete _den;
      if( _velx != 0 ) delete _velx;
      if( _vely != 0 ) delete _vely;
      if( _velz != 0 ) delete _velz;
      if( _frcx != 0 ) delete _frcx;
      if( _frcy != 0 ) delete _frcy;
      if( _frcz != 0 ) delete _frcz;
    }
    FieldData *den() const {return _den;}
    FieldData *velx() const {return _velx;}
    FieldData *vely() const {return _vely;}
    FieldData *velz() const {return _velz;}
    FieldData *frcx() const {return _frcx;}
    FieldData *frcy() const {return _frcy;}
    FieldData *frcz() const {return _frcz;}

    NodeSet *get_nodeset() const {return _nodes;}

    double get_iden() const {return _iden;}
    void get_ivel(double ivel[3]) const {
      ivel[0] = _ivel[0]; ivel[1] = _ivel[1]; ivel[2] = _ivel[2];
    }
    void get_ifrc(double ifrc[3]) const {
      ifrc[0] = _ifrc[0]; ifrc[1] = _ifrc[1]; ifrc[2] = _ifrc[2];
    }

  protected:
    NodeSet *_nodes;
    FieldData *_den;
    FieldData *_velx;
    FieldData *_vely;
    FieldData *_velz;
    FieldData *_frcx;
    FieldData *_frcy;
    FieldData *_frcz;
    double _iden, _ivel[3], _ifrc[3];
};
//---------------------------------------------------------------------------
// Class Geometry
//---------------------------------------------------------------------------
class Geometry
{
  public:
    Geometry(Lattice *lat) {
      _lat = lat;
      _phase = new GeomData(lat, SOLID_NODE);
    }
    ~Geometry() {if( _phase != 0 ) delete _phase;}

    Lattice *get_lattice() const {return _lat;}
    GeomData *get_phase() const {return _phase;}

  protected:
    Lattice *_lat;
    GeomData *_phase;
};
//===========================================================================
#endif
//===========================================================================