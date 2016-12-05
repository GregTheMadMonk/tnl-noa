/***************************************************************************
                          tnlSDFSinWaveFunctionSDFSDF_impl.h  -  description
                             -------------------
    begin                : Oct 13, 2014
    copyright            : (C) 2014 by Tomas Sobotik

 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#pragma once

#include <functions/tnlSinWaveFunctionSDF.h>

template< int dimensions, typename Real >
tnlSinWaveFunctionSDFBase< dimensions, Real >::tnlSinWaveFunctionSDFBase()
: waveLength( 1.0 ),
  amplitude( 1.0 ),
  phase( 0 ),
  wavesNumber( 0 )
{
}

template< int dimensions, typename Real >
bool tnlSinWaveFunctionSDFBase< dimensions, Real >::setup( const Config::ParameterContainer& parameters,
                                           const String& prefix )
{
   this->waveLength = parameters.getParameter< double >( prefix + "wave-length" );
   this->amplitude = parameters.getParameter< double >( prefix + "amplitude" );
   this->phase = parameters.getParameter< double >( prefix + "phase" );
   while(this->phase >2.0*M_PI)
      this->phase -= 2.0*M_PI;
   this->wavesNumber = ceil( parameters.getParameter< double >( prefix + "waves-number" ) );
   return true;
}

template< int dimensions, typename Real >
void tnlSinWaveFunctionSDFBase< dimensions, Real >::setWaveLength( const Real& waveLength )
{
   this->waveLength = waveLength;
}

template< int dimensions, typename Real >
Real tnlSinWaveFunctionSDFBase< dimensions, Real >::getWaveLength() const
{
   return this->waveLength;
}

template< int dimensions, typename Real >
void tnlSinWaveFunctionSDFBase< dimensions, Real >::setAmplitude( const Real& amplitude )
{
   this->amplitude = amplitude;
}

template< int dimensions, typename Real >
Real tnlSinWaveFunctionSDFBase< dimensions, Real >::getAmplitude() const
{
   return this->amplitude;
}

template< int dimensions, typename Real >
void tnlSinWaveFunctionSDFBase< dimensions, Real >::setPhase( const Real& phase )
{
   this->phase = phase;
}

template< int dimensions, typename Real >
Real tnlSinWaveFunctionSDFBase< dimensions, Real >::getPhase() const
{
   return this->phase;
}

template< int dimensions, typename Real >
void tnlSinWaveFunctionSDFBase< dimensions, Real >::setWavesNumber( const Real& wavesNumber )
{
   this->wavesNumber = wavesNumber;
}

template< int dimensions, typename Real >
Real tnlSinWaveFunctionSDFBase< dimensions, Real >::getWavesNumber() const
{
   return this->wavesNumber;
}

template< int dimensions, typename Real >
__cuda_callable__
Real tnlSinWaveFunctionSDFBase< dimensions, Real >::sinWaveFunctionSDF( const Real& r ) const
{
   if( this->wavesNumber == 0.0 || r < this->wavesNumber * this->waveLength )
      return Sign( r - round( 2.0 * r / this->waveLength ) * this->waveLength / 2.0 )
             * ( r - round( 2.0 * r / this->waveLength ) * this->waveLength / 2.0 )
             * Sign( sin( 2.0 * M_PI * r / this->waveLength ) );
   else
      return r - this->wavesNumber * this->waveLength;   
}


template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder>
__cuda_callable__
Real
tnlSinWaveFunctionSDF< 1, Real >::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   const RealType& x = v.x();
   const RealType distance = sqrt( x * x ) + this->phase * this->waveLength / (2.0*M_PI);
   if( XDiffOrder == 0 )
      return this->sinWaveFunctionSDF( distance );
   TNL_ASSERT( false, "TODO: implement this" );
   return 0.0;
}


template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder>
__cuda_callable__
Real
tnlSinWaveFunctionSDF< 2, Real >::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   if( ZDiffOrder != 0 )
      return 0.0;

   const RealType& x = v.x();
   const RealType& y = v.y();
   const RealType distance  = sqrt( x * x + y * y ) + this->phase * this->waveLength / (2.0*M_PI);
   if( XDiffOrder == 0 && YDiffOrder == 0)
      return this->sinWaveFunctionSDF( distance );
   TNL_ASSERT( false, "TODO: implement this" );
   return 0.0;
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder>
__cuda_callable__
Real
tnlSinWaveFunctionSDF< 3, Real >::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   const RealType& z = v.z();
   const RealType distance  = sqrt( x * x +  y * y + z * z ) +  this->phase * this->waveLength / (2.0*M_PI);
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
      return this->sinWaveFunctionSDF( distance );
   TNL_ASSERT( false, "TODO: implement this" );
   return 0.0;
}
