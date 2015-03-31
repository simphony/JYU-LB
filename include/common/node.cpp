//===========================================================================
// Author: Keijo Mattila, JYU, March 2015
// Description: 
// Details: 
//===========================================================================
#include "node.h"
//===========================================================================
// Class Lattice
//---------------------------------------------------------------------------
Lattice::Lattice(const Lattice *lat)
{
  _ncount = lat->_ncount;
  _size[0] = lat->_size[0];
  _size[1] = lat->_size[1];
  _size[2] = lat->_size[2];
  _origin[0] = lat->_origin[0];
  _origin[1] = lat->_origin[1];
  _origin[2] = lat->_origin[2];
}
//---------------------------------------------------------------------------
Lattice::Lattice(const UINT size[3], const double origin[3])
  throw(std::invalid_argument)
{
  if( size[0] == 0 || size[1] == 0 || size[2] == 0 ) {
    throw std::invalid_argument("Lattice size must be > 0!");
  }
  _ncount = size[0]*size[1]*size[2];
  _size[0] = size[0]; _size[1] = size[1]; _size[2] = size[2];
  _origin[0] = origin[0]; _origin[1] = origin[1]; _origin[2] = origin[2];
}
//---------------------------------------------------------------------------
Lattice::~Lattice()
{
}
//---------------------------------------------------------------------------
UINT Lattice::get_n(const UINT ijk[3]) const throw(std::out_of_range)
{
  #ifdef SAFE_MODE
  if( ijk[0] >= _size[0] || ijk[1] >= _size[1] || ijk[2] >= _size[2] ) {
    throw std::out_of_range("Index coordinates out of range!");
  }
  #endif
  return (ijk[2]*_size[0]*_size[1] + ijk[1]*_size[0] + ijk[0]);
}
//---------------------------------------------------------------------------
void Lattice::get_ijk(UINT n, UINT ijk[3]) const throw(std::out_of_range)
{
  #ifdef SAFE_MODE
  if( n >= _ncount ) {
    throw std::out_of_range("Enumeration number out of range!");
  }
  #endif
  UINT size_ij = _size[0]*_size[1], rem = n%size_ij;
  ijk[2] = n/size_ij;
  ijk[1] = rem/_size[0];
  ijk[0] = rem%_size[0];
}
//---------------------------------------------------------------------------
void Lattice::get_size(UINT size[3]) const
{
  size[0] = _size[0]; size[1] = _size[1]; size[2] = _size[2];
}
//---------------------------------------------------------------------------
void Lattice::get_origin(double origin[3]) const
{
  origin[0] = _origin[0]; origin[1] = _origin[1]; origin[2] = _origin[2];
}
//---------------------------------------------------------------------------
// Class NodeSubset
//---------------------------------------------------------------------------
NodeSubset::NodeSubset(NodeSet *superset, Filter *cond)
{
  _superset = superset;
  _store_node_count(cond);
  _store_maps(cond);
}
//---------------------------------------------------------------------------
NodeSubset::NodeSubset(NodeIterator *iter, Filter *cond)
{
  _superset = iter->get_nodeset();
  _store_node_count(iter, cond);
  _store_maps(iter, cond);
}
//---------------------------------------------------------------------------
NodeSubset::~NodeSubset()
{
  if( _is != 0 ) delete [] _is;
  if( _js != 0 ) delete [] _js;
  if( _ks != 0 ) delete [] _ks;
  if( _map_ijk2n != 0 ) delete [] _map_ijk2n;
}
//---------------------------------------------------------------------------
UINT NodeSubset::get_n(const UINT ijk[3]) const throw(std::out_of_range)
{
  UINT sn = _superset->get_n(ijk), n = _map_ijk2n[sn];
  #ifdef SAFE_MODE
  if( n == 0 ) {
    throw std::out_of_range("Index coordinates out of range!");
  }
  #endif
  return (n - 1);
}
//---------------------------------------------------------------------------
void NodeSubset::get_ijk(UINT n, UINT ijk[3]) const throw(std::out_of_range)
{
  #ifdef SAFE_MODE
  if( n >= _ncount ) {
    throw std::out_of_range("Enumeration number out of range!");
  }
  #endif
  ijk[0] = _is[n]; ijk[1] = _js[n]; ijk[2] = _ks[n];
}
//---------------------------------------------------------------------------
void NodeSubset::_store_node_count(Filter *cond)
{
  _ncount = 0;
  for(UINT sn = 0; sn < _superset->get_node_count(); ++sn) {
    UINT ijk[3];
    _superset->get_ijk(sn, ijk);
    if( cond->is_valid(ijk) ) ++_ncount;
  }
}
//---------------------------------------------------------------------------
void NodeSubset::_store_node_count(NodeIterator *iter, Filter *cond)
{
  _ncount = 0;
  for(UINT selem = 0; selem < iter->get_node_count(); ++selem) {
    UINT ijk[3];
    iter->get_ijk(selem, ijk);
    if( cond->is_valid(ijk) ) ++_ncount;
  }
}
//---------------------------------------------------------------------------
void NodeSubset::_store_maps(Filter *cond)
{
  _is = new UINT[_ncount];
  _js = new UINT[_ncount];
  _ks = new UINT[_ncount];

  UINT sncount = _superset->get_node_count();
  _map_ijk2n = new UINT[sncount];
  for(UINT sn = 0; sn < sncount; ++sn) {_map_ijk2n[sn] = 0;}

  UINT n = 0;
  for(UINT sn = 0; sn < sncount; ++sn) {
    UINT ijk[3];
    _superset->get_ijk(sn, ijk);
    if( !cond->is_valid(ijk) ) continue;

    _map_ijk2n[sn] = n+1;          
    _is[n] = ijk[0];
    _js[n] = ijk[1];
    _ks[n] = ijk[2];
    ++n;
  }
}
//---------------------------------------------------------------------------
void NodeSubset::_store_maps(NodeIterator *iter, Filter *cond)
{
  _is = new UINT[_ncount];
  _js = new UINT[_ncount];
  _ks = new UINT[_ncount];

  UINT sncount = _superset->get_node_count();
  _map_ijk2n = new UINT[sncount];
  for(UINT sn = 0; sn < sncount; ++sn) {_map_ijk2n[sn] = 0;}

  UINT n = 0;
  for(UINT selem = 0; selem < iter->get_node_count(); ++selem) {
    UINT ijk[3];
    iter->get_ijk(selem, ijk);
    UINT sn = iter->get_n(selem);
    if( !cond->is_valid(ijk) ) continue;

    _map_ijk2n[sn] = n+1;          
    _is[n] = ijk[0];
    _js[n] = ijk[1];
    _ks[n] = ijk[2];
    ++n;
  }
}
//---------------------------------------------------------------------------
// Class NodeIterator
//---------------------------------------------------------------------------
NodeIterator::NodeIterator(NodeSet *nodes, Filter *cond)
{
  _nodes = nodes;
  _store_node_count(cond);
  _store_valid_nodes(cond);
}
//---------------------------------------------------------------------------
NodeIterator::NodeIterator(NodeIterator *iter, Filter *cond)
{
  _nodes = iter->get_nodeset();
  _store_node_count(iter, cond);
  _store_valid_nodes(iter, cond);
}
//---------------------------------------------------------------------------
NodeIterator::~NodeIterator()
{
  if( _ns != 0 ) delete [] _ns;
}
//---------------------------------------------------------------------------
void NodeIterator::get_ijk(UINT elem, UINT ijk[3]) const
  throw(std::out_of_range)
{
  #ifdef SAFE_MODE
  if( elem >= _ncount ) {
    throw std::out_of_range("Element number out of range!");
  }
  #endif
  _nodes->get_ijk(_ns[elem], ijk);
}
//---------------------------------------------------------------------------
UINT NodeIterator::get_n(UINT elem) const throw(std::out_of_range)
{
  #ifdef SAFE_MODE
  if( elem >= _ncount ) {
    throw std::out_of_range("Element number out of range!");
  }
  #endif
  return _ns[elem];
}
//---------------------------------------------------------------------------
void NodeIterator::_store_node_count(Filter *cond)
{
  _ncount = 0;
  for(UINT n = 0; n < _nodes->get_node_count(); ++n) {
    UINT ijk[3];
    _nodes->get_ijk(n, ijk);
    if( cond->is_valid(ijk) ) ++_ncount;
  }
}
//---------------------------------------------------------------------------
void NodeIterator::_store_node_count(NodeIterator *iter, Filter *cond)
{
  _ncount = 0;
  for(UINT selem = 0; selem < iter->get_node_count(); ++selem) {
    UINT ijk[3];
    iter->get_ijk(selem, ijk);
    if( cond->is_valid(ijk) ) ++_ncount;
  }
}
//---------------------------------------------------------------------------
void NodeIterator::_store_valid_nodes(Filter *cond)
{
  _ns = new UINT[_ncount];

  UINT elem = 0;
  for(UINT n = 0; n < _nodes->get_node_count(); ++n) {
    UINT ijk[3];
    _nodes->get_ijk(n, ijk);
    if( !cond->is_valid(ijk) ) continue;

    _ns[elem] = n;
    ++elem;
  }
}
//---------------------------------------------------------------------------
void NodeIterator::_store_valid_nodes(NodeIterator *iter, Filter *cond)
{
  _ns = new UINT[_ncount];

  UINT elem = 0;
  for(UINT selem = 0; selem < iter->get_node_count(); ++selem) {
    UINT ijk[3];
    iter->get_ijk(selem, ijk);
    UINT n = iter->get_n(selem);
    if( !cond->is_valid(ijk) ) continue;

    _ns[elem] = n;
    ++elem;
  }
}
//===========================================================================