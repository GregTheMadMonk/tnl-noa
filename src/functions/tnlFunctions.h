/* 
 * File:   tnlFunctions.h
 * Author: oberhuber
 *
 * Created on July 11, 2016, 6:01 PM
 */

#pragma once

#include <core/tnlCuda.h>

template< typename Real >
__cuda_callable__
Real sign( const Real& x, const Real& smoothing = 0.0 )
{
   if( x > smoothing )
      return 1.0;
   else if( x < -smoothing )
      return -1.0;
   if( smoothing == 0.0 )
      return 0.0;
   return sin( ( M_PI * x ) / ( 2.0 * smoothing ) );
}

template< typename Real >
__cuda_callable__
Real positivePart( const Real& arg)
{
   return arg > 0.0 ? arg : 0.0;
}

template< typename Real >
__cuda_callable__
Real negativePart( const Real& arg)
{
   return arg < 0.0 ? arg : 0.0;
}

template< typename Real >
__cuda_callable__
Real ArgAbsMin( const Real& x, const Real& y )
{
   return fabs( x ) < fabs( y ) ?  x : y;
}

template< typename Real >
__cuda_callable__
Real ArgAbsMax( const Real& x, const Real& y )
{
   return fabs( x ) > fabs( y ) ?  x : y;
}

