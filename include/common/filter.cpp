//===========================================================================
// Author: Keijo Mattila, JYU, March 2015
// Description: 
// Details: 
//===========================================================================
#include "filter.h"
#include "data.h"
//===========================================================================
// Class CrossSectionFilter
//---------------------------------------------------------------------------
CrossSectionFilter::CrossSectionFilter(UINT axis, UINT index)
{
  _axis = axis;
  _index = index;
}
//---------------------------------------------------------------------------
CrossSectionFilter::~CrossSectionFilter()
{
}
//---------------------------------------------------------------------------
bool CrossSectionFilter::is_valid(const UINT ijk[3]) const
{
  return ijk[_axis] == _index;
}
//---------------------------------------------------------------------------
// Class DomainFilter
//---------------------------------------------------------------------------
DomainFilter::DomainFilter(UINT ijk_from[3], UINT ijk_to[3])
{
  _ijk_from[0] = ijk_from[0];
  _ijk_from[1] = ijk_from[1];
  _ijk_from[2] = ijk_from[2];

  _ijk_to[0] = ijk_to[0];
  _ijk_to[1] = ijk_to[1];
  _ijk_to[2] = ijk_to[2];
}
//---------------------------------------------------------------------------
DomainFilter::~DomainFilter()
{
}
//---------------------------------------------------------------------------
bool DomainFilter::is_valid(const UINT ijk[3]) const
{
  return ijk[0] >= _ijk_from[0] && ijk[0] < _ijk_to[0] &&
         ijk[1] >= _ijk_from[1] && ijk[1] < _ijk_to[1] &&
         ijk[2] >= _ijk_from[2] && ijk[2] < _ijk_to[2];
}
//---------------------------------------------------------------------------
// Class DataValueFilter
//---------------------------------------------------------------------------
template <class T>
DataValueFilter<T>::DataValueFilter(NodeData<T> *ndata, T value)
{
  _ndata = ndata;
  _value = value;
}
//---------------------------------------------------------------------------
template <class T>
DataValueFilter<T>::~DataValueFilter()
{
}
//---------------------------------------------------------------------------
template <class T>
bool DataValueFilter<T>::is_valid(const UINT ijk[3]) const
{
  T data_val = _ndata->get_val_ijk(ijk);
  return (data_val == _value);
}
//===========================================================================
template class DataValueFilter<unsigned char>;
template class DataValueFilter<double>;
//===========================================================================