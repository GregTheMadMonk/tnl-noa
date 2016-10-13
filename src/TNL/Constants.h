/***************************************************************************
                           tnlConstants.h -  description
                             -------------------
    begin                : June 17, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <limits.h>
#include <float.h>
#include <TNL/Assert.h>
#include <TNL/Devices/Cuda.h>

namespace TNL {

template< typename T > constexpr T MinValue() { return T();};
template<> constexpr char               MinValue< char >() { return CHAR_MIN; }
template<> constexpr unsigned char      MinValue< unsigned char >() { return 0; }
template<> constexpr short int          MinValue< short int >() { return SHRT_MIN; }
template<> constexpr unsigned short int MinValue< unsigned short int >() { return 0; }
template<> constexpr int                MinValue< int >() { return INT_MIN; }
template<> constexpr unsigned int       MinValue< unsigned int >() { return 0; }
template<> constexpr long int           MinValue< long int >() { return LONG_MIN; }
template<> constexpr unsigned long int  MinValue< unsigned long int >() { return 0; }
template<> constexpr float              MinValue< float >() { return -FLT_MAX; }
template<> constexpr double             MinValue< double >() { return -DBL_MAX; }
template<> constexpr long double        MinValue< long double >() { return -LDBL_MAX; }

template< typename T > constexpr T MaxValue() { return T();};
template<> constexpr char               MaxValue< char >() { return CHAR_MAX; }
template<> constexpr unsigned char      MaxValue< unsigned char >() { return UCHAR_MAX; }
template<> constexpr short int          MaxValue< short int >() { return SHRT_MAX; }
template<> constexpr unsigned short int MaxValue< unsigned short int >() { return USHRT_MAX; }
template<> constexpr int                MaxValue< int >() { return INT_MAX; }
template<> constexpr unsigned int       MaxValue< unsigned int >() { return UINT_MAX; }
template<> constexpr long int           MaxValue< long int >() { return LONG_MAX; }
template<> constexpr unsigned long int  MaxValue< unsigned long int >() { return ULONG_MAX; }
template<> constexpr float              MaxValue< float >() { return FLT_MAX; }
template<> constexpr double             MaxValue< double >() { return DBL_MAX; }
template<> constexpr long double        MaxValue< long double >() { return LDBL_MAX; }

} // namespace TNL

