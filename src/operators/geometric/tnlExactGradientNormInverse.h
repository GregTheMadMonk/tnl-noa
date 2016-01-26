/***************************************************************************
                          tnlExactGradientNormInverse.h  -  description
                             -------------------
    begin                : Jan 21, 2016
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

#ifndef TNLEXACTGRADIENTTNORMINVERSE_H
#define	TNLEXACTGRADIENTTNORMINVERSE_H

#include <core/vectors/tnlVector.h>
#include <core/vectors/tnlSharedVector.h>
#include <mesh/tnlGrid.h>
#include <functions/tnlDomain.h>
#include <operators/geometric/tnlExactGradientNorm.h>

template< typename ExactGradientNorm,
          int Dimensions = ExactGradientNorm::getDimensions() >
class tnlExactGradientNormInverse
{};

/****
 * 1D
 */
template< typename ExactGradientNorm >
class tnlExactGradientNormInverse< ExactGradientNorm, 1 >
   : public tnlDomain< 1, SpaceDomain >
{
   public:

      static tnlString getType()
      {
         return "tnlExactGradientNormInverse< 1 >";
      }
      
      tnlExactGradientNormInverse(){};

      void setRegularization( const Real& epsilon )
      {
         this->gradientNorm.setEps( epsilon );
      }
      
      template< typename Function >
      __cuda_callable__
      typename Function::RealType 
         operator()( const Function& function,
                     const typename Function::VertexType& v, 
                     const typename Function::RealType& time = 0.0 ) const
      {
         return 1.0 / gradientNorm( function, v, time );
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
         static_assert( XDerivative < 2, "Partial derivative of higher order then 1 are not implemented yet." );
         typedef typename Function::RealType RealType;
         
         if( XDerivative == 1 )
         {
            const RealType q = gradientNorm( function, v, time );
            const RealType q_x = gradientNorm.template getPartialDerivative< Function, 1, 0, 0 >( function, v, time );            
            return - q_x / q;         
         }
         if( XDerivative == 0 )
            return this->operator()( function, v, time );         
         if( YDerivative != 0 || ZDerivative != 0 )
            return 0.0;         
      }
      
      protected:
         
         ExactGradientNorm gradientNorm;      
};


/****
 * 2D
 */
template< typename ExactGradientNorm >
class tnlExactGradientNormInverse< ExactGradientNorm, 2 >
   : public tnlDomain< 2, SpaceDomain >
{
   public:

      static tnlString getType()
      {
         return "tnlExactGradientNormInverse< 2 >";
      }
      
      tnlExactGradientNormInverse(){};

      void setRegularization( const Real& epsilon )
      {
         this->gradientNorm.setEps( epsilon );
      }
      
      template< typename Function >
      __cuda_callable__
      typename Function::RealType 
         operator()( const Function& function,
                     const typename Function::VertexType& v, 
                     const typename Function::RealType& time = 0.0 ) const
      {
         return 1.0 / gradientNorm( function, v, time );
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
         static_assert( XDerivative < 2 && YDerivative < 2, "Partial derivative of higher order then 1 are not implemented yet." );
         typedef typename Function::RealType RealType;
         
         if( XDerivative == 1 && YDerivative == 0 )
         {
            const RealType q = gradientNorm( function, v, time );
            const RealType q_x = gradientNorm.template getPartialDerivative< Function, 1, 0, 0 >( function, v, time );            
            return - q_x / q;         
         }
         if( XDerivative == 0 && YDerivative == 1 )
         {
            const RealType q = gradientNorm( function, v, time );
            const RealType q_y = gradientNorm.template getPartialDerivative< Function, 0, 1, 0 >( function, v, time );            
            return - q_y / q;                     
         }         
         if( XDerivative == 0 && YDerivative == 0 )
            return this->operator()( function, v, time );         
         if( ZDerivative > 0 )
            return 0.0;         
      }
      
      protected:
         
         Real epsilonSquare;      
};

template< typename ExactGradientNorm >
class tnlExactGradientNormInverse< ExactGradientNorm, 3 >
   : public tnlDomain< 3, SpaceDomain >
{
   public:

      static tnlString getType()
      {
         return "tnlExactGradientNormInverse< 3 >";
      }
      
      tnlExactGradientNormInverse(){};

      void setRegularization( const Real& epsilon )
      {
         this->gradientNorm.setEps( epsilon );
      }
      
      template< typename Function >
      __cuda_callable__
      typename Function::RealType 
         operator()( const Function& function,
                     const typename Function::VertexType& v, 
                     const typename Function::RealType& time = 0.0 ) const
      {
         return 1.0 / gradientNorm( function, v, time );
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
         static_assert( XDerivative < 2 && YDerivative < 2 && ZDerivative < 2,
            "Partial derivative of higher order then 1 are not implemented yet." );

         typedef typename Function::RealType RealType;
         if( XDerivative == 1 && YDerivative == 0 && ZDerivative == 0 )
         {
            const RealType q = gradientNorm( function, v, time );
            const RealType q_x = gradientNorm.template getPartialDerivative< Function, 1, 0, 0 >( function, v, time );            
            return - q_x / q;                     
         }
         if( XDerivative == 0 && YDerivative == 1 && ZDerivative == 0 )
         {
            const RealType q = gradientNorm( function, v, time );
            const RealType q_y = gradientNorm.template getPartialDerivative< Function, 0, 1, 0 >( function, v, time );            
            return - q_y / q;                     
         }         
         if( XDerivative == 0 && YDerivative == 0 && ZDerivative == 1 )
         {
            const RealType q = gradientNorm( function, v, time );
            const RealType q_z = gradientNorm.template getPartialDerivative< Function, 0, 0, 1 >( function, v, time );            
            return - q_z / q;                     
         }
         if( XDerivative == 0 && YDerivative == 0 && ZDerivative == 0 )
            return this->operator()( function, v, time );                  
      }
      
      protected:
         
         Real epsilonSquare;      
};

#endif	/* TNLEXACTGRADIENTTNORM_H */
