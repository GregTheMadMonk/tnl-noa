/***************************************************************************
                          ExactIdentityOperator.h  -  description
                             -------------------
    begin                : Feb 18, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/String.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Operators/Operator.h>

namespace TNL {
namespace Operators {

template< int Dimension >
class ExactIdentityOperator
   : public Functions::Domain< Dimension, Functions::SpaceDomain >
{
   public:
 
      static String getType()
      {
         return String( "ExactIdentityOperator< " ) +
                String( Dimension) + " >";
      }
 
      template< typename Function >
      __cuda_callable__
      typename Function::RealType
         operator()( const Function& function,
                     const typename Function::PointType& v,
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
                               const typename Function::PointType& v,
                               const typename Function::RealType& time = 0.0 ) const
      {
         static_assert( XDerivative >= 0 && YDerivative >= 0 && ZDerivative >= 0,
            "Partial derivative must be non-negative integer." );
 
         return function.template getPartialDerivative< XDerivative, YDerivative, ZDerivative >( v, time );
      }
};

} // namespace Operators
} // namespace TNL

