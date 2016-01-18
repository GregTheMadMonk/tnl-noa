/***************************************************************************
                          tnlExactGradientNorm.h  -  description
                             -------------------
    begin                : Jan 18, 2016
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

#ifndef TNLEXACTGRADIENTTNORM_H
#define	TNLEXACTGRADIENTTNORM_H

#include <core/vectors/tnlVector.h>
#include <core/vectors/tnlSharedVector.h>
#include <mesh/tnlGrid.h>
#include <functions/tnlDomain.h>

template< int Dimensions,
          typename Real = double >
class tnlExactGradientNorm
{};

/****
 * 1D
 */
template< typename Real >
class tnlExactGradientNorm< 1, Real >
   : public tnlDomain< 1, SpaceDomain >
{
   public:

      static tnlString getType()
      {
         return "tnlExactGradientNorm< 1 >";
      }
      
      tnlExactGradientNorm()
      : epsilonSquare( 1.0 ){};

      void setRegularization( const Real& epsilon )
      {
         this->epsilonSquare = epsilon*epsilon;
      }
      
      template< typename Function >
      __cuda_callable__
      typename Function::RealType 
         operator()( const Function& function,
                     const typename Function::VertexType& v, 
                     const typename Function::RealType& time = 0.0 ) const
      {
         typedef typename Function::RealType RealType;
         const RealType f_x = function.template getPartialDerivative< 1, 0, 0 >( v, time );
         return sqrt( this->epsilonSquare + f_x * f_x );         
      }
      
      template< typename Function, 
                int XDerivative = 0,
                int YDerivative = 0,
                int ZDerivative = 0, 
                typename Real = typename Function::RealType >
      __cuda_callable__
      typename Function::RealType
         getPartialDerivative( const Function& function,
                               const typename Function::Vertex& v,
                               const typename Function::RealType& time = 0.0 ) const
      {
         static_assert( XDerivative >= 0 && YDerivative >= 0 && ZDerivative >= 0,
            "Partial derivative must be non-negative integer." );
         static_assert( XDerivative < 2, "Partial derivative of higher order then 1 are not implemented yet." );
         typedef typename Function::RealType ReealType;
         
         if( YDerivative != 0 || ZDerivative != 0 )
            return 0.0;
         if( XDerivative == 0 )
            return this->operator()( function, vertex, time );
         if( XDerivative == 1 )
         {
            const RealType f_x = function.template getPartialDerivative< 1, 0, 0 >( v, time );
            const RealType f_xx = function.template getPartialDerivative< 2, 0, 0 >( v, time ); 
            const RealType Q = sqrt( this->epsilonSquare + f_x * f_x );
            return ( f_x * f_xx ) / Q;         
         }
      }
      
      protected:
         
         Real epsilonSquare;      
};


/****
 * 2D
 */
template< typename Real >
class tnlExactGradientNorm< 2, Real >
   : public tnlDomain< 2, SpaceDomain >
{
   public:

      static tnlString getType()
      {
         return "tnlExactGradientNorm< 2 >";
      }
      
      tnlExactGradientNorm()
      : epsilonSquare( 1.0 ){};

      void setRegularization( const Real& epsilon )
      {
         this->epsilonSquare = epsilon*epsilon;
      }
      
      template< typename Function >
      __cuda_callable__
      typename Function::RealType 
         operator()( const Function& function,
                     const typename Function::VertexType& v, 
                     const typename Function::RealType& time = 0.0 ) const
      {
         
      }
      
      template< typename Function, 
                int XDerivative = 0,
                int YDerivative = 0,
                int ZDerivative = 0, 
                typename Real = typename Function::RealType >
      __cuda_callable__
      typename Function::RealType
         getPartialDerivative( const Function& function,
                               const typename Function::Vertex& v,
                               const Real& time = 0.0,
                               const Real& eps = 1.0 ) const
      {
         
      }
      
      protected:
         
         Real epsilonSquare;      
};

template< typename Real >
class tnlExactGradientNorm< 3, Real >
   : public tnlDomain< 3, SpaceDomain >
{
   public:

      static tnlString getType()
      {
         return "tnlExactGradientNorm< 3 >";
      }
      
      tnlExactGradientNorm()
      : epsilonSquare( 1.0 ){};

      void setRegularization( const Real& epsilon )
      {
         this->epsilonSquare = epsilon*epsilon;
      }
      
      template< typename Function >
      __cuda_callable__
      typename Function::RealType 
         operator()( const Function& function,
                     const typename Function::VertexType& v, 
                     const typename Function::RealType& time = 0.0 ) const
      {
      }
      
      template< typename Function, 
                int XDerivative = 0,
                int YDerivative = 0,
                int ZDerivative = 0, 
                typename Real = typename Function::RealType >
      __cuda_callable__
      typename Function::RealType
         getPartialDerivative( const Function& function,
                               const typename Function::Vertex& v,
                               const Real& time = 0.0 ) const
      {
      }
      
      protected:
         
         Real epsilonSquare;      
};


#include <operators/operator-Q/tnlExactGradientNorm_impl.h>


#endif	/* TNLEXACTGRADIENTTNORM_H */
