/***************************************************************************
                          StaticArray_impl.cpp  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Arrays/StaticArray.h>

namespace TNL {
namespace Arrays {    

#ifndef HAVE_CUDA
#ifdef UNDEF //TEMPLATE_EXPLICIT_INSTANTIATION

template class StaticArray< 1, char >;
template class StaticArray< 1, int >;
#ifdef INSTANTIATE_LONG_INT
template class StaticArray< 1, long int >;
#endif
#ifdef INSTANTIATE_FLOAT
template class StaticArray< 1, float >;
#endif
template class StaticArray< 1, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class StaticArray< 1, long double >;
#endif

template class StaticArray< 2, char >;
template class StaticArray< 2, int >;
#ifdef INSTANTIATE_LONG_INT
template class StaticArray< 2, long int >;
#endif
#ifdef INSTANTIATE_FLOAT
template class StaticArray< 2, float >;
#endif
template class StaticArray< 2, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class StaticArray< 2, long double >;
#endif

template class StaticArray< 3, char >;
template class StaticArray< 3, int >;
#ifdef INSTANTIATE_LONG_INT
template class StaticArray< 3, long int >;
#endif
#ifdef INSTANTIATE_FLOAT
template class StaticArray< 3, float >;
#endif
template class StaticArray< 3, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class StaticArray< 3, long double >;
#endif

template class StaticArray< 4, char >;
template class StaticArray< 4, int >;
#ifdef INSTANTIATE_LONG_INT
template class StaticArray< 4, long int >;
#endif
#ifdef INSTANTIATE_FLOAT
template class StaticArray< 4, float >;
#endif
template class StaticArray< 4, double >;
#ifdef INSTANTIATE_LONG_DOUBLE
template class StaticArray< 4, long double >;
#endif

#endif
#endif

} // namespace Arrays
} // namespace TNL

