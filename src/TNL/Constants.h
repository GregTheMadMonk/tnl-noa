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

template< typename T > constexpr T tnlMinValue() { Assert( false, ); return T(); };
template<> constexpr char               tnlMinValue< char >() { return CHAR_MIN; }
template<> constexpr unsigned char      tnlMinValue< unsigned char >() { return 0; }
template<> constexpr short int          tnlMinValue< short int >() { return SHRT_MIN; }
template<> constexpr unsigned short int tnlMinValue< unsigned short int >() { return 0; }
template<> constexpr int                tnlMinValue< int >() { return INT_MIN; }
template<> constexpr unsigned int       tnlMinValue< unsigned int >() { return 0; }
template<> constexpr long int           tnlMinValue< long int >() { return LONG_MIN; }
template<> constexpr unsigned long int  tnlMinValue< unsigned long int >() { return 0; }
template<> constexpr float              tnlMinValue< float >() { return -FLT_MAX; }
template<> constexpr double             tnlMinValue< double >() { return -DBL_MAX; }
template<> constexpr long double        tnlMinValue< long double >() { return -LDBL_MAX; }

template< typename T > constexpr T tnlMaxValue() { Assert( false, ); return T(); };
template<> constexpr char               tnlMaxValue< char >() { return CHAR_MAX; }
template<> constexpr unsigned char      tnlMaxValue< unsigned char >() { return UCHAR_MAX; }
template<> constexpr short int          tnlMaxValue< short int >() { return SHRT_MAX; }
template<> constexpr unsigned short int tnlMaxValue< unsigned short int >() { return USHRT_MAX; }
template<> constexpr int                tnlMaxValue< int >() { return INT_MAX; }
template<> constexpr unsigned int       tnlMaxValue< unsigned int >() { return UINT_MAX; }
template<> constexpr long int           tnlMaxValue< long int >() { return LONG_MAX; }
template<> constexpr unsigned long int  tnlMaxValue< unsigned long int >() { return ULONG_MAX; }
template<> constexpr float              tnlMaxValue< float >() { return FLT_MAX; }
template<> constexpr double             tnlMaxValue< double >() { return DBL_MAX; }
template<> constexpr long double        tnlMaxValue< long double >() { return LDBL_MAX; }

} // namespace TNL

