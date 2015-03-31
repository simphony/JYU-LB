//===========================================================================
// Author: Keijo Mattila, JYU, March 2015
// Description: 
// Details: 
//===========================================================================
#ifndef NODE_H
#define NODE_H
//===========================================================================
#include <stdexcept>
#include "filter.h"
//===========================================================================
// Class NodeSet
//---------------------------------------------------------------------------
class NodeSet
{
  public:
    NodeSet() {_ncount = 0;}
    virtual ~NodeSet() {}

    virtual UINT get_n(const UINT ijk[3]) const throw(std::out_of_range) = 0;
    virtual void get_ijk(UINT n, UINT ijk[3]) const
      throw(std::out_of_range) = 0;
      
    UINT get_node_count() const {return _ncount;}

  protected:
    UINT _ncount;
};
//---------------------------------------------------------------------------
// Class Lattice
//---------------------------------------------------------------------------
class Lattice: public NodeSet
{
  public:
    explicit Lattice(const Lattice *lat);
    explicit Lattice(const UINT size[3], const double origin[3])
      throw(std::invalid_argument);
    ~Lattice();

    UINT get_n(const UINT ijk[3]) const throw(std::out_of_range);
    void get_ijk(UINT n, UINT ijk[3]) const throw(std::out_of_range);

    void get_size(UINT size[3]) const;
    void get_origin(double origin[3]) const;

  protected:
    UINT _size[3];
    double _origin[3];
};
//---------------------------------------------------------------------------
// Class NodeSubset
//---------------------------------------------------------------------------
class NodeIterator;
class NodeSubset: public NodeSet
{
  public:
    explicit NodeSubset(NodeSet *superset, Filter *cond);
    explicit NodeSubset(NodeIterator *iter, Filter *cond);
    ~NodeSubset();

    UINT get_n(const UINT ijk[3]) const throw(std::out_of_range);
    void get_ijk(UINT n, UINT ijk[3]) const throw(std::out_of_range);
    NodeSet *get_superset() const {return _superset;}

  protected:
    NodeSet *_superset;
    UINT *_is, *_js, *_ks, *_map_ijk2n;

  private:
    void _store_node_count(Filter *cond);
    void _store_node_count(NodeIterator *iter, Filter *cond);
    void _store_maps(Filter *cond);
    void _store_maps(NodeIterator *iter, Filter *cond);
};
//---------------------------------------------------------------------------
// Class NodeIterator
//---------------------------------------------------------------------------
class NodeIterator
{
  public:
    explicit NodeIterator(NodeSet *nodes, Filter *cond);
    explicit NodeIterator(NodeIterator *iter, Filter *cond);
    ~NodeIterator();
     
    UINT get_n(UINT elem) const throw(std::out_of_range);
    void get_ijk(UINT elem, UINT ijk[3]) const throw(std::out_of_range);

    UINT get_node_count() const {return _ncount;}
    NodeSet *get_nodeset() const {return _nodes;}

  protected:
    NodeSet *_nodes;
    UINT _ncount, *_ns;

  private:
    void _store_node_count(Filter *cond);
    void _store_node_count(NodeIterator *iter, Filter *cond);
    void _store_valid_nodes(Filter *cond);
    void _store_valid_nodes(NodeIterator *iter, Filter *cond);
};
//===========================================================================
#endif
//===========================================================================