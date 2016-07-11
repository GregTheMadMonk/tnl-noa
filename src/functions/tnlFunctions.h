/* 
 * File:   tnlFunctions.h
 * Author: oberhuber
 *
 * Created on July 11, 2016, 6:01 PM
 */

#ifndef TNLFUNCTIONS_H
#define	TNLFUNCTIONS_H

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

#endif	/* TNLFUNCTIONS_H */

