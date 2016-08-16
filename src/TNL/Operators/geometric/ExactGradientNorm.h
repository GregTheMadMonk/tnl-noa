/***************************************************************************
                          ExactGradientNorm.h  -  description
                             -------------------
    begin                : Jan 18, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/SharedVector.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Functions/Domain.h>

namespace TNL {
namespace Operators {   

template< int Dimensions,
          typename Real = double >
class ExactGradientNorm
{};

/****
 * 1D
 */
template< typename Real >
class ExactGradientNorm< 1, Real >
   : public Functions::Domain< 1, Functions::SpaceDomain >
{
   public:

      static String getType()
      {
         return "ExactGradientNorm< 1 >";
      }
 
      ExactGradientNorm()
      : epsilonSquare( 0.0 ){};

      void setRegularizationEpsilon( const Real& epsilon )
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
         return ::sqrt( this->epsilonSquare + f_x * f_x );
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
            const RealType f_x = function.template getPartialDerivative< 1, 0, 0 >( v, time );
            const RealType f_xx = function.template getPartialDerivative< 2, 0, 0 >( v, time );
            const RealType Q = ::sqrt( this->epsilonSquare + f_x * f_x );
            return ( f_x * f_xx ) / Q;
         }
         if( XDerivative == 0 )
            return this->operator()( function, v, time );
         if( YDerivative != 0 || ZDerivative != 0 )
            return 0.0;
      }
 
      protected:
 
         Real epsilonSquare;
};


/****
 * 2D
 */
template< typename Real >
class ExactGradientNorm< 2, Real >
   : public Functions::Domain< 2, Functions::SpaceDomain >
{
   public:

      static String getType()
      {
         return "ExactGradientNorm< 2 >";
      }
 
      ExactGradientNorm()
      : epsilonSquare( 0.0 ){};

      void setRegularizationEpsilon( const Real& epsilon )
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
         const RealType f_y = function.template getPartialDerivative< 0, 1, 0 >( v, time );
         return ::sqrt( this->epsilonSquare + f_x * f_x + f_y * f_y );
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
            const RealType f_x  = function.template getPartialDerivative< 1, 0, 0 >( v, time );
            const RealType f_y  = function.template getPartialDerivative< 0, 1, 0 >( v, time );
            const RealType f_xx = function.template getPartialDerivative< 2, 0, 0 >( v, time );
            const RealType f_xy = function.template getPartialDerivative< 1, 1, 0 >( v, time );
            return ( f_x *  f_xx + f_y * f_xy ) / ::sqrt( this->epsilonSquare + f_x * f_x + f_y * f_y );
         }
         if( XDerivative == 0 && YDerivative == 1 )
         {
            const RealType f_x  = function.template getPartialDerivative< 1, 0, 0 >( v, time );
            const RealType f_y  = function.template getPartialDerivative< 0, 1, 0 >( v, time );
            const RealType f_xy = function.template getPartialDerivative< 1, 1, 0 >( v, time );
            const RealType f_yy = function.template getPartialDerivative< 0, 2, 0 >( v, time );
            return ( f_x *  f_xy + f_y * f_yy ) / ::sqrt( this->epsilonSquare + f_x * f_x + f_y * f_y );
         }
         if( XDerivative == 0 && YDerivative == 0 )
            return this->operator()( function, v, time );
         if( ZDerivative > 0 )
            return 0.0;
      }
 
      protected:
 
         Real epsilonSquare;
};

template< typename Real >
class ExactGradientNorm< 3, Real >
   : public Functions::Domain< 3, Functions::SpaceDomain >
{
   public:

      static String getType()
      {
         return "ExactGradientNorm< 3 >";
      }
 
      ExactGradientNorm()
      : epsilonSquare( 0.0 ){};

      void setRegularizationEpsilon( const Real& epsilon )
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
         const RealType f_y = function.template getPartialDerivative< 0, 1, 0 >( v, time );
         const RealType f_z = function.template getPartialDerivative< 0, 0, 1 >( v, time );
         return std::sqrt( this->epsilonSquare + f_x * f_x + f_y * f_y + f_z * f_z );
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
            const RealType f_x  = function.template getPartialDerivative< 1, 0, 0 >( v, time );
            const RealType f_y  = function.template getPartialDerivative< 0, 1, 0 >( v, time );
            const RealType f_z  = function.template getPartialDerivative< 0, 0, 1 >( v, time );
            const RealType f_xx = function.template getPartialDerivative< 2, 0, 0 >( v, time );
            const RealType f_xy = function.template getPartialDerivative< 1, 1, 0 >( v, time );
            const RealType f_xz = function.template getPartialDerivative< 1, 0, 1 >( v, time );
            return ( f_x *  f_xx + f_y * f_xy + f_z * f_xz ) /
               std::sqrt( this->epsilonSquare + f_x * f_x + f_y * f_y + f_z * f_z );
         }
         if( XDerivative == 0 && YDerivative == 1 && ZDerivative == 0 )
         {
            const RealType f_x  = function.template getPartialDerivative< 1, 0, 0 >( v, time );
            const RealType f_y  = function.template getPartialDerivative< 0, 1, 0 >( v, time );
            const RealType f_z  = function.template getPartialDerivative< 0, 0, 1 >( v, time );
            const RealType f_xy = function.template getPartialDerivative< 1, 1, 0 >( v, time );
            const RealType f_yy = function.template getPartialDerivative< 0, 2, 0 >( v, time );
            const RealType f_yz = function.template getPartialDerivative< 0, 1, 1 >( v, time );
            return ( f_x *  f_xy + f_y * f_yy + f_z * f_yz ) /
               std::sqrt( this->epsilonSquare + f_x * f_x + f_y * f_y + f_z * f_z );
         }
         if( XDerivative == 0 && YDerivative == 0 && ZDerivative == 1 )
         {
            const RealType f_x  = function.template getPartialDerivative< 1, 0, 0 >( v, time );
            const RealType f_y  = function.template getPartialDerivative< 0, 1, 0 >( v, time );
            const RealType f_z  = function.template getPartialDerivative< 0, 0, 1 >( v, time );
            const RealType f_xz = function.template getPartialDerivative< 1, 0, 1 >( v, time );
            const RealType f_yz = function.template getPartialDerivative< 0, 1, 1 >( v, time );
            const RealType f_zz = function.template getPartialDerivative< 0, 0, 2 >( v, time );
            return ( f_x *  f_xz + f_y * f_yz + f_z * f_zz ) /
               std::sqrt( this->epsilonSquare + f_x * f_x + f_y * f_y + f_z * f_z );
         }
         if( XDerivative == 0 && YDerivative == 0 && ZDerivative == 0 )
            return this->operator()( function, v, time );
      }
 
      protected:
 
         Real epsilonSquare;
};

} // namespace Operators
} // namespace TNL