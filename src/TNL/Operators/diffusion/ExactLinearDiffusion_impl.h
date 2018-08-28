/***************************************************************************
                          ExactLinearDiffusion_impl.h  -  description
                             -------------------
    begin                : Aug 8, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Szekely Ondrej, ondra.szekely@gmail.com
 */

#pragma once

namespace TNL {
namespace Operators {

String
ExactLinearDiffusion< 1 >::
getType()
{
   return "ExactLinearDiffusion< 1 >";
}

template< typename Function >
__cuda_callable__ inline
typename Function::RealType
ExactLinearDiffusion< 1 >::
operator()( const Function& function,
            const typename Function::PointType& v,
            const typename Function::RealType& time ) const
{
   return function.template getPartialDerivative< 2, 0, 0 >( v, time );
}

String
ExactLinearDiffusion< 2 >::
getType()
{
   return "ExactLinearDiffusion< 2 >";
}

template< typename Function >
__cuda_callable__ inline
typename Function::RealType
ExactLinearDiffusion< 2 >::
operator()( const Function& function,
            const typename Function::PointType& v,
          const typename Function::RealType& time ) const
{
   return function.template getPartialDerivative< 2, 0, 0 >( v, time ) +
          function.template getPartialDerivative< 0, 2, 0 >( v, time );
}

String
ExactLinearDiffusion< 3 >::
getType()
{
   return "ExactLinearDiffusion< 3 >";
}

template< typename Function >
__cuda_callable__ inline
typename Function::RealType
ExactLinearDiffusion< 3 >::
operator()( const Function& function,
            const typename Function::PointType& v,
            const typename Function::RealType& time ) const
{
   return function.template getPartialDerivative< 2, 0, 0 >( v, time ) +
          function.template getPartialDerivative< 0, 2, 0 >( v, time ) +
          function.template getPartialDerivative< 0, 0, 2 >( v, time );

}

} // namespace Operators
} // namespace TNL
