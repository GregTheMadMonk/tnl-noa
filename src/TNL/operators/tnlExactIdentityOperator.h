/***************************************************************************
                          tnlExactIdentityOperator.h  -  description
                             -------------------
    begin                : Feb 18, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/core/tnlString.h>
#include <TNL/core/tnlCuda.h>
#include <TNL/operators/tnlOperator.h>

namespace TNL {

template< int Dimensions >
class tnlExactIdentityOperator
   : public tnlDomain< Dimensions, SpaceDomain >
{
   public:
 
      static tnlString getType()
      {
         return tnlString( "tnlExactIdentityOperator< " ) +
                tnlString( Dimensions) + " >";
      }
 
      template< typename Function >
      __cuda_callable__
      typename Function::RealType
         operator()( const Function& function,
                     const typename Function::VertexType& v,
                     const typename Function::RealType& time = 0.0 ) const
      {
         return function( v, time );
      }
 
      template< typename Function,
                int XDerivative = 0,
                int YDerivative = 0,
                int ZDerivative = 0 >
      __cuda_callable__
      typename Function::RealType
         getPartialDerivative( const Function& function,
                               const typename Function::VertexType& v,
                               const typename Function::RealType& time = 0.0 ) const
      {
         static_assert( XDerivative >= 0 && YDerivative >= 0 && ZDerivative >= 0,
            "Partial derivative must be non-negative integer." );
 
         return function.template getPartialDerivative< XDerivative, YDerivative, ZDerivative >( v, time );
      }
};

} // namespace TNL

