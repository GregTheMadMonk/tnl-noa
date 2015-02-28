/***************************************************************************
                          tnlExpBumpFunction_impl.h  -  description
                             -------------------
    begin                : Dec 5, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef TNLEXPBUMPFUNCTION_IMPL_H_
#define TNLEXPBUMPFUNCTION_IMPL_H_

#include <functions/tnlExpBumpFunction.h>

template< typename Real >
bool
tnlExpBumpFunctionBase< Real >::
setup( const tnlParameterContainer& parameters,
       const tnlString& prefix )
{
   this->amplitude = parameters.getParameter< double >( prefix + "amplitude" );
   this->sigma = parameters.getParameter< double >( prefix + "sigma" );
   return true;
}

template< typename Real >
void tnlExpBumpFunctionBase< Real >::setAmplitude( const Real& amplitude )
{
   this->amplitude = amplitude;
}

template< typename Real >
const Real& tnlExpBumpFunctionBase< Real >::getAmplitude() const
{
   return this->amplitude;
}

template< typename Real >
void tnlExpBumpFunctionBase< Real >::setSigma( const Real& sigma )
{
   this->sigma = sigma;
}

template< typename Real >
const Real& tnlExpBumpFunctionBase< Real >::getSigma() const
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
   return "tnlExpBumpFunction< 1, " + ::getType< Real >() + tnlString( " >" );
}

template< typename Real >
tnlExpBumpFunction< 1, Real >::tnlExpBumpFunction()
{
}

template< typename Real >
   template< int XDiffOrder, 
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
Real
tnlExpBumpFunction< 1, Real >::getValue( const Vertex& v,
                                         const Real& time ) const
{
   const RealType& x = v.x();
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
      return this->amplitude * exp( -x*x / ( this->sigma*this->sigma ) );
   if( XDiffOrder == 1 )
      return -2.0 * x / ( this->sigma * this->sigma ) * this->amplitude * exp( -x*x / ( this->sigma * this->sigma ) );
   if( XDiffOrder == 2 )
      return -2.0 / ( this->sigma * this->sigma ) * this->amplitude * exp( -x*x / ( this->sigma * this->sigma ) ) + 4.0 * x * x / ( this->sigma * this->sigma * this->sigma * this->sigma ) * this->amplitude * exp( -x*x / ( this->sigma * this->sigma ) );
   return 0.0;
}

/****
 * 2D
 */

template< typename Real >
tnlString
tnlExpBumpFunction< 2, Real >::getType()
{
   return tnlString( "tnlExpBumpFunction< 2, " ) + ::getType< Real >() + " >";
}

template< typename Real >
tnlExpBumpFunction< 2, Real >::tnlExpBumpFunction()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
Real
tnlExpBumpFunction< 2, Real >::
getValue( const Vertex& v,
          const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   if( ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 && YDiffOrder == 0 )
      return this->amplitude * exp( ( -x*x - y*y ) / ( this->sigma * this->sigma ) );
   if( XDiffOrder == 1 && YDiffOrder == 0 )
      return -2.0 * x / ( this->sigma * this->sigma ) * this->amplitude * exp( (-x * x - y * y)/ ( this->sigma * this->sigma ) );
   if( XDiffOrder == 2 && YDiffOrder == 0 )
      return -2.0 / ( this->sigma * this->sigma ) * this->amplitude * exp( (-x*x - y*y) / ( this->sigma * this->sigma ) ) + 4.0 * x * x / ( this->sigma * this->sigma * this->sigma * this->sigma ) * this->amplitude * exp( (-x*x - y*y) / ( this->sigma * this->sigma ) );
   if( XDiffOrder == 0 && YDiffOrder == 1 )
      return -2.0 * y / ( this->sigma * this->sigma ) * this->amplitude * exp( (-x * x - y * y)/ ( this->sigma * this->sigma ) );
   if( XDiffOrder == 0 && YDiffOrder == 2 )
      return -2.0 / ( this->sigma * this->sigma ) * this->amplitude * exp( (-x*x - y*y) / ( this->sigma * this->sigma ) ) + 4.0 * y * y / ( this->sigma * this->sigma * this->sigma * this->sigma ) * this->amplitude * exp( (-x*x - y*y) / ( this->sigma * this->sigma ) );
   return 0.0;
}

/****
 * 3D
 */

template< typename Real >
tnlString
tnlExpBumpFunction< 3, Real >::getType()
{
   return tnlString( "tnlExpBumpFunction< 3, " ) + ::getType< Real >() + " >";
}

template< typename Real >
tnlExpBumpFunction< 3, Real >::tnlExpBumpFunction()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
Real
tnlExpBumpFunction< 3, Real >::
getValue( const Vertex& v,
          const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   const RealType& z = v.z();
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
      return this->amplitude * exp( ( -x*x - y*y -z*z ) / ( this->sigma * this->sigma ) );
   if( XDiffOrder == 1 && YDiffOrder == 0 && ZDiffOrder == 0 )
      return -2.0 * x / ( this->sigma * this->sigma ) * this->amplitude * exp( ( -x*x - y*y -z*z ) / ( this->sigma * this->sigma ) );
   if( XDiffOrder == 2 && YDiffOrder == 0 && ZDiffOrder == 0 )
      return -2.0 / ( this->sigma * this->sigma ) * this->amplitude * exp( ( -x*x - y*y -z*z ) / ( this->sigma * this->sigma ) ) + 4.0 * x * x / ( this->sigma * this->sigma * this->sigma * this->sigma ) * this->amplitude * exp( ( -x*x - y*y -z*z ) / ( this->sigma * this->sigma ) );
   if( XDiffOrder == 0 && YDiffOrder == 1 && ZDiffOrder == 0 )
      return -2.0 * y / ( this->sigma * this->sigma ) * this->amplitude * exp( ( -x*x - y*y -z*z ) / ( this->sigma * this->sigma ) );
   if( XDiffOrder == 0 && YDiffOrder == 2 && ZDiffOrder == 0 )
      return -2.0 / ( this->sigma * this->sigma ) * this->amplitude * exp( ( -x*x - y*y -z*z ) / ( this->sigma * this->sigma ) ) + 4.0 * y * y / ( this->sigma * this->sigma * this->sigma * this->sigma ) * this->amplitude * exp( ( -x*x - y*y -z*z ) / ( this->sigma * this->sigma ) );
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 1 )
      return -2.0 * z / ( this->sigma * this->sigma ) * this->amplitude * exp( ( -x*x - y*y -z*z ) / ( this->sigma * this->sigma ) );
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 2 )
      return -2.0 / ( this->sigma * this->sigma ) * this->amplitude * exp( ( -x*x - y*y -z*z ) / ( this->sigma * this->sigma ) ) + 4.0 * z * z / ( this->sigma * this->sigma * this->sigma * this->sigma ) * this->amplitude * exp( ( -x*x - y*y -z*z ) / ( this->sigma * this->sigma ) );
   return 0.0;
}


#endif /* TNLEXPBUMPFUNCTION_IMPL_H_ */