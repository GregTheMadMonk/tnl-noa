/***************************************************************************
                          tnlExactMeanCurvature.h  -  description
                             -------------------
    begin                : Feb 18, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/operators/diffusion/tnlExactNonlinearDiffusion.h>
#include <TNL/operators/tnlExactFunctionInverseOperator.h>
#include <TNL/operators/geometric/tnlExactGradientNorm.h>

namespace TNL {

template< int Dimensions,
          typename InnerOperator = tnlExactIdentityOperator< Dimensions > >
class tnlExactMeanCurvature
: public Functions::Domain< Dimensions, Functions::SpaceDomain >
{
   public:
 
      typedef tnlExactGradientNorm< Dimensions > ExactGradientNorm;
      typedef tnlExactFunctionInverseOperator< Dimensions, ExactGradientNorm > FunctionInverse;
      typedef tnlExactNonlinearDiffusion< Dimensions, FunctionInverse > NonlinearDiffusion;
 
      static String getType()
      {
         return String( "tnlExactMeanCurvature< " ) +
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
 
      ExactGradientNorm gradientNorm;
 
      FunctionInverse functionInverse;
 
      NonlinearDiffusion nonlinearDiffusion;
 
};

} // namespace TNL

