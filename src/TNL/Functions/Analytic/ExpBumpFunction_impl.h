/***************************************************************************
                          ExpBumpFunction_impl.h  -  description
                             -------------------
    begin                : Dec 5, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <cmath>
#include <TNL/Functions/Analytic/ExpBumpFunction.h>

namespace TNL {
namespace Functions {
namespace Analytic {   

template< int dimensions, typename Real >
ExpBumpFunctionBase< dimensions, Real >::
ExpBumpFunctionBase()
   : amplitude( 1.0 ), sigma( 1.0 )
{
}

template< int dimensions, typename Real >
bool
ExpBumpFunctionBase< dimensions, Real >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   this->amplitude = parameters.getParameter< double >( prefix + "amplitude" );
   this->sigma = parameters.getParameter< double >( prefix + "sigma" );
   return true;
}

template< int dimensions, typename Real >
void ExpBumpFunctionBase< dimensions, Real >::setAmplitude( const Real& amplitude )
{
   this->amplitude = amplitude;
}

template< int dimensions, typename Real >
const Real& ExpBumpFunctionBase< dimensions, Real >::getAmplitude() const
{
   return this->amplitude;
}

template< int dimensions, typename Real >
void ExpBumpFunctionBase< dimensions, Real >::setSigma( const Real& sigma )
{
   this->sigma = sigma;
}

template< int dimensions, typename Real >
const Real& ExpBumpFunctionBase< dimensions, Real >::getSigma() const
{
   return this->sigma;
}

/***
 * 1D
 */

template< typename Real >
String
ExpBumpFunction< 1, Real >::getType()
{
   return "ExpBumpFunction< 1, " + TNL::getType< Real >() + String( " >" );
}

template< typename Real >
ExpBumpFunction< 1, Real >::ExpBumpFunction()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__
Real
ExpBumpFunction< 1, Real >::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   using namespace std;
   const RealType& x = v.x();
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
      return this->amplitude * ::exp( -x*x / ( this->sigma*this->sigma ) );
   if( XDiffOrder == 1 )
      return -2.0 * x / ( this->sigma * this->sigma ) * this->amplitude * ::exp( -x*x / ( this->sigma * this->sigma ) );
   if( XDiffOrder == 2 )
      return -2.0 / ( this->sigma * this->sigma ) * this->amplitude * ::exp( -x*x / ( this->sigma * this->sigma ) ) + 4.0 * x * x / ( this->sigma * this->sigma * this->sigma * this->sigma ) * this->amplitude * ::exp( -x*x / ( this->sigma * this->sigma ) );
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
ExpBumpFunction< 1, Real >::
operator()( const VertexType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}


/****
 * 2D
 */

template< typename Real >
String
ExpBumpFunction< 2, Real >::getType()
{
   return String( "ExpBumpFunction< 2, " ) + TNL::getType< Real >() + " >";
}

template< typename Real >
ExpBumpFunction< 2, Real >::ExpBumpFunction()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__ inline
Real
ExpBumpFunction< 2, Real >::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   if( ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 && YDiffOrder == 0 )
      return this->amplitude * ::exp( ( -x*x - y*y ) / ( this->sigma * this->sigma ) );
   if( XDiffOrder == 1 && YDiffOrder == 0 )
      return -2.0 * x / ( this->sigma * this->sigma ) * this->amplitude * ::exp( (-x * x - y * y)/ ( this->sigma * this->sigma ) );
   if( XDiffOrder == 2 && YDiffOrder == 0 )
      return -2.0 / ( this->sigma * this->sigma ) * this->amplitude * ::exp( (-x*x - y*y) / ( this->sigma * this->sigma ) ) + 4.0 * x * x / ( this->sigma * this->sigma * this->sigma * this->sigma ) * this->amplitude * ::exp( (-x*x - y*y) / ( this->sigma * this->sigma ) );
   if( XDiffOrder == 0 && YDiffOrder == 1 )
      return -2.0 * y / ( this->sigma * this->sigma ) * this->amplitude * ::exp( (-x * x - y * y)/ ( this->sigma * this->sigma ) );
   if( XDiffOrder == 0 && YDiffOrder == 2 )
      return -2.0 / ( this->sigma * this->sigma ) * this->amplitude * ::exp( (-x*x - y*y) / ( this->sigma * this->sigma ) ) + 4.0 * y * y / ( this->sigma * this->sigma * this->sigma * this->sigma ) * this->amplitude * ::exp( (-x*x - y*y) / ( this->sigma * this->sigma ) );
   if( XDiffOrder == 1 && YDiffOrder == 1 )
      return 4.0 * x * y / ( ( this->sigma * this->sigma ) * ( this->sigma * this->sigma ) ) * this->amplitude * ::exp( (-x * x - y * y)/ ( this->sigma * this->sigma ) );
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
ExpBumpFunction< 2, Real >::
operator()( const VertexType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}

/****
 * 3D
 */

template< typename Real >
String
ExpBumpFunction< 3, Real >::getType()
{
   return String( "ExpBumpFunction< 3, " ) + TNL::getType< Real >() + " >";
}

template< typename Real >
ExpBumpFunction< 3, Real >::ExpBumpFunction()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__
Real
ExpBumpFunction< 3, Real >::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   const RealType& z = v.z();
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
      return this->amplitude * ::exp( ( -x*x - y*y -z*z ) / ( this->sigma * this->sigma ) );
   if( XDiffOrder == 1 && YDiffOrder == 0 && ZDiffOrder == 0 )
      return -2.0 * x / ( this->sigma * this->sigma ) * this->amplitude * ::exp( ( -x*x - y*y -z*z ) / ( this->sigma * this->sigma ) );
   if( XDiffOrder == 2 && YDiffOrder == 0 && ZDiffOrder == 0 )
      return -2.0 / ( this->sigma * this->sigma ) * this->amplitude * ::exp( ( -x*x - y*y -z*z ) / ( this->sigma * this->sigma ) ) + 4.0 * x * x / ( this->sigma * this->sigma * this->sigma * this->sigma ) * this->amplitude * ::exp( ( -x*x - y*y -z*z ) / ( this->sigma * this->sigma ) );
   if( XDiffOrder == 0 && YDiffOrder == 1 && ZDiffOrder == 0 )
      return -2.0 * y / ( this->sigma * this->sigma ) * this->amplitude * ::exp( ( -x*x - y*y -z*z ) / ( this->sigma * this->sigma ) );
   if( XDiffOrder == 0 && YDiffOrder == 2 && ZDiffOrder == 0 )
      return -2.0 / ( this->sigma * this->sigma ) * this->amplitude * ::exp( ( -x*x - y*y -z*z ) / ( this->sigma * this->sigma ) ) + 4.0 * y * y / ( this->sigma * this->sigma * this->sigma * this->sigma ) * this->amplitude * ::exp( ( -x*x - y*y -z*z ) / ( this->sigma * this->sigma ) );
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 1 )
      return -2.0 * z / ( this->sigma * this->sigma ) * this->amplitude * ::exp( ( -x*x - y*y -z*z ) / ( this->sigma * this->sigma ) );
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 2 )
      return -2.0 / ( this->sigma * this->sigma ) * this->amplitude * ::exp( ( -x*x - y*y -z*z ) / ( this->sigma * this->sigma ) ) + 4.0 * z * z / ( this->sigma * this->sigma * this->sigma * this->sigma ) * this->amplitude * ::exp( ( -x*x - y*y -z*z ) / ( this->sigma * this->sigma ) );
   if( XDiffOrder == 1 && YDiffOrder == 1 && ZDiffOrder == 0 )
      return 4.0 * x * y / ( ( this->sigma * this->sigma ) * ( this->sigma * this->sigma ) ) * this->amplitude * ::exp( ( -x*x - y*y -z*z ) / ( this->sigma * this->sigma ) );
   if( XDiffOrder == 1 && YDiffOrder == 0 && ZDiffOrder == 1 )
      return 4.0 * x * z / ( ( this->sigma * this->sigma ) * ( this->sigma * this->sigma ) ) * this->amplitude * ::exp( ( -x*x - y*y -z*z ) / ( this->sigma * this->sigma ) );
   if( XDiffOrder == 0 && YDiffOrder == 1 && ZDiffOrder == 1 )
      return 4.0 * y * z / ( ( this->sigma * this->sigma ) * ( this->sigma * this->sigma ) ) * this->amplitude * ::exp( ( -x*x - y*y -z*z ) / ( this->sigma * this->sigma ) );
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
ExpBumpFunction< 3, Real >::
operator()( const VertexType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}

} // namespace Analytic
} // namespace Functions
} // namespace TNL
