/***************************************************************************
                          tnlExactMeanCurvature.h  -  description
                             -------------------
    begin                : Feb 18, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/


#ifndef TNLEXACTMEANCURVATURE_H
#define	TNLEXACTMEANCURVATURE_H

#include<operators/diffusion/tnlExactNonlinearDiffusion.h>
#include<operators/tnlExactFunctionInverseOperator.h>
#include<operators/geometric/tnlExactGradientNorm.h>

template< int Dimensions,
          typename InnerOperator = tnlExactIdentityOperator< Dimensions > >
class tnlExactMeanCurvature
: public tnlDomain< Dimensions, SpaceDomain >
{
   public:
     
      typedef tnlExactGradientNorm< Dimensions > ExactGradientNorm;
      typedef tnlExactFunctionInverseOperator< Dimensions, ExactGradientNorm > FunctionInverse;
      typedef tnlExactNonlinearDiffusion< Dimensions, FunctionInverse > NonlinearDiffusion;
      
      static tnlString getType()
      {
         return tnlString( "tnlExactMeanCurvature< " ) + 
                tnlString( Dimensions) + ", " +
                InnerOperator::getType() + " >";         
      }
      
      template< typename Real >
      void setRegularizationEpsilon( const Real& eps)
      {
         nonlinearDiffusion.getNonlinearity().getInnerOperator().setRegularizationEpislon( eps );
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


#endif	/* TNLEXACTMEANCURVATURE_H */
