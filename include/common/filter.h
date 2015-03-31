//===========================================================================
// Author: Keijo Mattila, JYU, March 2015
// Description: 
// Details: 
//===========================================================================
#ifndef FILTER_H
#define FILTER_H
//===========================================================================
#include "def.h"
//===========================================================================
// Class Filter
//---------------------------------------------------------------------------
class Filter
{
  public:
    Filter() {}
    virtual ~Filter() {}
    virtual bool is_valid(const UINT ijk[3]) const = 0;
};
//---------------------------------------------------------------------------
// Class CrossSectionFilter
//---------------------------------------------------------------------------
class CrossSectionFilter: public Filter
{
  public:
    CrossSectionFilter(UINT axis, UINT index);
    ~CrossSectionFilter();
    bool is_valid(const UINT ijk[3]) const;
  protected:
    UINT _axis, _index;
};
//---------------------------------------------------------------------------
// Class DomainFilter
//---------------------------------------------------------------------------
class DomainFilter: public Filter
{
  public:
    DomainFilter(UINT ijk_from[3], UINT ijk_to[3]);
    ~DomainFilter();
    bool is_valid(const UINT ijk[3]) const;
  protected:
    UINT _ijk_from[3], _ijk_to[3];
};
//---------------------------------------------------------------------------
// Class DataValueFilter
//---------------------------------------------------------------------------
template <class T> class NodeData;
template <class T> class DataValueFilter: public Filter
{
  public:
    DataValueFilter(NodeData<T> *ndata, T value);
    ~DataValueFilter();
    bool is_valid(const UINT ijk[3]) const;
  protected:
    NodeData<T> *_ndata;
    T _value;
};
//===========================================================================
#endif
//===========================================================================