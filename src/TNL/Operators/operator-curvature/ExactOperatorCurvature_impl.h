/***************************************************************************
                          ExactLinearDiffusion_impl.h  -  description
                             -------------------
    begin                : Aug 8, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Operators/operator-curvature/ExactOperatorCurvature.h>

namespace TNL {
namespace Operators {   

template< typename ExactOperatorQ >
String
ExactOperatorCurvature< ExactOperatorQ, 1 >::
getType()
{
   return "ExactOperatorCurvature< " + ExactOperatorQ::getType() + ",1 >";
}

template< typename OperatorQ >
template< int XDiffOrder, int YDiffOrder, int ZDiffOrder, typename Function, typename Point, typename Real >
__cuda_callable__
Real
tnlExactOperatorQ< 1 >::
getValue( const Function& function,
          const Point& v,
          const Real& time, const Real& eps )
{
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
        return 0.0;
   if (XDiffOrder == 0)
        return function.template getValue< 2, 0, 0, Point >( v, time )/ExactOperatorQ::template getValue< 0, 0, 0 >( function, v, time, eps ) -
               ( function.template getValue< 1, 0, 0, Point >( v, time ) * ExactOperatorQ::template getValue< 1, 0, 0 >( function, v, time, eps ) )
                / ( ExactOperatorQ::template getValue< 0, 0, 0 >( function, v, time, eps ) * ExactOperatorQ::template getValue< 0, 0, 0 >( function, v, time, eps ) );
   return 0;
}

template< typename ExactOperatorQ >
String
ExactOperatorCurvature< ExactOperatorQ, 2 >::
getType()
{
   return "ExactOperatorCurvature< " + ExactOperatorQ::getType() + ",2 >";
}

template< int XDiffOrder, int YDiffOrder, int ZDiffOrder, typename Function, typename Point, typename Real >
__cuda_callable__
Real
tnlExactOperatorQ< 2 >::
getValue( const Function& function,
          const Point& v,
          const Real& time, const Real& eps )
{
   if( ZDiffOrder != 0 )
        return 0.0;
   if (XDiffOrder == 0 && YDiffOrder == 0 )
        return ( function.template getValue< 2, 0, 0, Point >( v, time ) * function.template getValue< 0, 2, 0, Point >( v, time ) )
               /ExactOperatorQ::template getValue< 0, 0, 0 >( function, v, time, eps ) - ( function.template getValue< 1, 0, 0, Point >( v, time ) *
               ExactOperatorQ::template getValue< 1, 0, 0 >( function, v, time, eps ) + function.template getValue< 0, 1, 0, Point >( v, time ) *
               ExactOperatorQ::template getValue< 0, 1, 0 >( function, v, time, eps ) )
                / ( ExactOperatorQ::template getValue< 0, 0, 0 >( function, v, time, eps ) * ExactOperatorQ::template getValue< 0, 0, 0 >( function, v, time, eps ) );
   return 0;
}

template< typename ExactOperatorQ >
String
ExactOperatorCurvature< ExactOperatorQ, 3 >::
getType()
{
   return "ExactOperatorCurvature< " + ExactOperatorQ::getType() + ",3 >";
}

} // namespace Operators
} // namespace TNL
