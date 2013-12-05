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

#include <generators/functions/tnlExpBumpFunction.h>

template< typename Real >
bool tnlExpBumpFunctionBase< Real >::init( const tnlParameterContainer& parameters )
{
   this->amplitude = parameters.GetParameter< double >( "amplitude" );
   this->sigma = parameters.GetParameter< double >( "sigma" );
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

template< typename Vertex, typename Device >
tnlExpBumpFunction< 1, Vertex, Device >::tnlExpBumpFunction()
{
}

template< typename Vertex, typename Device >
   template< int XDiffOrder, int YDiffOrder, int ZDiffOrder >
      typename Vertex::RealType tnlExpBumpFunction< 1, Vertex, Device >::getF( const Vertex& v ) const
{
   const RealType& x = v.x();
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
      return this->amplitude * sin( -x*x / ( this->sigma*this->sigma ) );
   return 0.0;
}

/****
 * 2D
 */

template< typename Vertex, typename Device >
tnlExpBumpFunction< 2, Vertex, Device >::tnlExpBumpFunction()
{
}

template< typename Vertex, typename Device >
   template< int XDiffOrder, int YDiffOrder, int ZDiffOrder >
      typename Vertex::RealType tnlExpBumpFunction< 2, Vertex, Device >::getF( const Vertex& v ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
      return this->amplitude * exp( ( -x*x - y*y ) / ( this->sigma * this->sigma ) );
   return 0.0;
}

/****
 * 3D
 */

template< typename Vertex, typename Device >
tnlExpBumpFunction< 3, Vertex, Device >::tnlExpBumpFunction()
{
}

template< typename Vertex, typename Device >
   template< int XDiffOrder, int YDiffOrder, int ZDiffOrder >
      typename Vertex::RealType tnlExpBumpFunction< 3, Vertex, Device >::getF( const Vertex& v ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   const RealType& z = v.z();
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
      return this->amplitude * exp( ( -x*x - y*y -z*z ) / ( this->sigma * this->sigma ) );
   return 0.0;
}

#endif /* TNLEXPBUMPFUNCTION_IMPL_H_ */
