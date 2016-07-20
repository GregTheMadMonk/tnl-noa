/***************************************************************************
                          tnlExactLinearDiffusion_impl.h  -  description
                             -------------------
    begin                : Aug 8, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

String
tnlExactLinearDiffusion< 1 >::
getType()
{
   return "tnlExactLinearDiffusion< 1 >";
}

template< typename Function >
__cuda_callable__ inline
typename Function::RealType
tnlExactLinearDiffusion< 1 >::
operator()( const Function& function,
            const typename Function::VertexType& v,
            const typename Function::RealType& time ) const
{
   return function.template getPartialDerivative< 2, 0, 0 >( v, time );
}

String
tnlExactLinearDiffusion< 2 >::
getType()
{
   return "tnlExactLinearDiffusion< 2 >";
}

template< typename Function >
__cuda_callable__ inline
typename Function::RealType
tnlExactLinearDiffusion< 2 >::
operator()( const Function& function,
            const typename Function::VertexType& v,
          const typename Function::RealType& time ) const
{
   return function.template getPartialDerivative< 2, 0, 0 >( v, time ) +
          function.template getPartialDerivative< 0, 2, 0 >( v, time );
}

String
tnlExactLinearDiffusion< 3 >::
getType()
{
   return "tnlExactLinearDiffusion< 3 >";
}

template< typename Function >
__cuda_callable__ inline
typename Function::RealType
tnlExactLinearDiffusion< 3 >::
operator()( const Function& function,
            const typename Function::VertexType& v,
            const typename Function::RealType& time ) const
{
   return function.template getPartialDerivative< 2, 0, 0 >( v, time ) +
          function.template getPartialDerivative< 0, 2, 0 >( v, time ) +
          function.template getPartialDerivative< 0, 0, 2 >( v, time );

}

} // namespace TNL
