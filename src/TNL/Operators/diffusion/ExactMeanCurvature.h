/***************************************************************************
                          ExactMeanCurvature.h  -  description
                             -------------------
    begin                : Feb 18, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Operators/diffusion/ExactNonlinearDiffusion.h>
#include <TNL/Operators/ExactFunctionInverseOperator.h>
#include <TNL/Operators/geometric/ExactGradientNorm.h>

namespace TNL {
namespace Operators {   

template< int Dimensions,
          typename InnerOperator = ExactIdentityOperator< Dimensions > >
class ExactMeanCurvature
: public Functions::Domain< Dimensions, Functions::SpaceDomain >
{
   public:
 
      typedef ExactGradientNorm< Dimensions > ExactGradientNormType;
      typedef ExactFunctionInverseOperator< Dimensions, ExactGradientNormType > FunctionInverse;
      typedef ExactNonlinearDiffusion< Dimensions, FunctionInverse > NonlinearDiffusion;
 
      static String getType()
      {
         return String( "ExactMeanCurvature< " ) +
                String( Dimensions) + ", " +
                InnerOperator::getType() + " >";
      }
 
      template< typename Real >
      void setRegularizationEpsilon( const Real& eps)
      {
         nonlinearDiffusion.getNonlinearity().getInnerOperator().setRegularizationEpsilon( eps );
      }
 
      template< typename Function >
      __cuda_callable__
      typename Function::RealType
         operator()( const Function& function,
                     const typename Function::VertexType& v,
                     const typename Function::RealType& time = 0.0 ) const
      {
         return this->nonlinearDiffusion( function, v, time );
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
         static_assert( XDerivative + YDerivative + ZDerivative < 1, "Partial derivative of higher order then 1 are not implemented yet." );
         typedef typename Function::RealType RealType;
 
         if( XDerivative == 1 )
         {
         }
         if( YDerivative == 1 )
         {
         }
         if( ZDerivative == 1 )
         {
         }
      }
 
 
   protected:
 
      ExactGradientNormType gradientNorm;
 
      FunctionInverse functionInverse;
 
      NonlinearDiffusion nonlinearDiffusion;
 
};

} // namespace Operators
} // namespace TNL

