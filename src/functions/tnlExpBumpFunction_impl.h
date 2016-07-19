/***************************************************************************
                          tnlExpBumpFunction_impl.h  -  description
                             -------------------
    begin                : Dec 5, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <cmath>
#include <functions/tnlExpBumpFunction.h>

namespace TNL {

template< int dimensions, typename Real >
tnlExpBumpFunctionBase< dimensions, Real >::
tnlExpBumpFunctionBase()
   : amplitude( 1.0 ), sigma( 1.0 )
{
}

template< int dimensions, typename Real >
bool
tnlExpBumpFunctionBase< dimensions, Real >::
setup( const tnlParameterContainer& parameters,
       const tnlString& prefix )
{
   this->amplitude = parameters.getParameter< double >( prefix + "amplitude" );
   this->sigma = parameters.getParameter< double >( prefix + "sigma" );
   return true;
}

template< int dimensions, typename Real >
void tnlExpBumpFunctionBase< dimensions, Real >::setAmplitude( const Real& amplitude )
{
   this->amplitude = amplitude;
}

template< int dimensions, typename Real >
const Real& tnlExpBumpFunctionBase< dimensions, Real >::getAmplitude() const
{
   return this->amplitude;
}

template< int dimensions, typename Real >
void tnlExpBumpFunctionBase< dimensions, Real >::setSigma( const Real& sigma )
{
   this->sigma = sigma;
}

template< int dimensions, typename Real >
const Real& tnlExpBumpFunctionBase< dimensions, Real >::getSigma() const
{
   return this->sigma;
}

/***
 * 1D
 */

template< typename Real >
tnlString
tnlExpBumpFunction< 1, Real >::getType()
{
   return "tnlExpBumpFunction< 1, " + TNL::getType< Real >() + tnlString( " >" );
}

template< typename Real >
tnlExpBumpFunction< 1, Real >::tnlExpBumpFunction()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__
Real
tnlExpBumpFunction< 1, Real >::
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
tnlExpBumpFunction< 1, Real >::
operator()( const VertexType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}


/****
 * 2D
 */

template< typename Real >
tnlString
tnlExpBumpFunction< 2, Real >::getType()
{
   return tnlString( "tnlExpBumpFunction< 2, " ) + TNL::getType< Real >() + " >";
}

template< typename Real >
tnlExpBumpFunction< 2, Real >::tnlExpBumpFunction()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__ inline
Real
tnlExpBumpFunction< 2, Real >::
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
tnlExpBumpFunction< 2, Real >::
operator()( const VertexType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}

/****
 * 3D
 */

template< typename Real >
tnlString
tnlExpBumpFunction< 3, Real >::getType()
{
   return tnlString( "tnlExpBumpFunction< 3, " ) + TNL::getType< Real >() + " >";
}

template< typename Real >
tnlExpBumpFunction< 3, Real >::tnlExpBumpFunction()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__
Real
tnlExpBumpFunction< 3, Real >::
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
tnlExpBumpFunction< 3, Real >::
operator()( const VertexType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}

} //namespace TNL
