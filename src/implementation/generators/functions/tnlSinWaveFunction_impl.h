/***************************************************************************
                          tnlSinWaveFunction_impl.h  -  description
                             -------------------
    begin                : Nov 19, 2013
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

#ifndef TNLSINWAVEFUNCTION_IMPL_H_
#define TNLSINWAVEFUNCTION_IMPL_H_

#include <generators/functions/tnlSinWaveFunction.h>

template< typename Real >
tnlSinWaveFunctionBase< Real >::tnlSinWaveFunctionBase()
: waveLength( 0 ),
  amplitude( 0 ),
  phase( 0 ),
  wavesNumber( 0 )
{
}

template< typename Real >
bool tnlSinWaveFunctionBase< Real >::init( const tnlParameterContainer& parameters )
{
   this->waveLength = parameters.GetParameter< double >( "wave-length" );
   this->amplitude = parameters.GetParameter< double >( "amplitude" );
   this->phase = parameters.GetParameter< double >( "phase" );
   parameters.GetParameter< double >( "waves-number" );
   return true;
}

template< typename Real >
void tnlSinWaveFunctionBase< Real >::setWaveLength( const Real& waveLength )
{
   this->waveLength = waveLength;
}

template< typename Real >
Real tnlSinWaveFunctionBase< Real >::getWaveLength() const
{
   return this->waveLength;
}

template< typename Real >
void tnlSinWaveFunctionBase< Real >::setAmplitude( const Real& amplitude )
{
   this->amplitude = amplitude;
}

template< typename Real >
Real tnlSinWaveFunctionBase< Real >::getAmplitude() const
{
   return this->amplitude;
}

template< typename Real >
void tnlSinWaveFunctionBase< Real >::setPhase( const Real& phase )
{
   this->phase = phase;
}

template< typename Real >
Real tnlSinWaveFunctionBase< Real >::getPhase() const
{
   return this->phase;
}

template< typename Vertex, typename Device >
   template< int XDiffOrder, int YDiffOrder, int ZDiffOrder >
      typename Vertex::RealType tnlSinWaveFunction< 1, Vertex, Device >::getF( const Vertex& v ) const
{
   const RealType& x = v.x();
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
   {
      RealType arg = 2.0 * M_PI * x  / this->waveLength;
      if( this->wavesNumber )
      {
         if( tnlAbs( arg ) > this->wavesNumber )
            arg = Sign( x ) * this->wavesNumber;
      }
      return this->amplitude * sin( this->phase + arg );
   }
   if( XDiffOrder == 1 )
      return 2.0 * M_PI / this->waveLength * this->amplitude * cos( this->phase + 2.0 * M_PI * x / this->waveLength );
   return 0.0;
}


template< typename Vertex, typename Device >
   template< int XDiffOrder, int YDiffOrder, int ZDiffOrder >
      typename Vertex::RealType tnlSinWaveFunction< 2, Vertex, Device >::getF( const VertexType& v ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
   {
      return this->amplitude * sin( this->phase + 2.0 * M_PI * sqrt( x * x + y * y ) / this->waveLength );
   }
   return 0.0;
}

template< typename Vertex, typename Device >
   template< int XDiffOrder, int YDiffOrder, int ZDiffOrder >
      typename Vertex::RealType tnlSinWaveFunction< 3, Vertex, Device >::getF( const VertexType& v ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   const RealType& z = v.z();
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
   {
      return this->amplitude * sin( this->phase + 2.0 * M_PI * sqrt( x * x + y * y + z * z ) / this->waveLength );
   }
   return 0.0;
}

#endif /* TNLSINWAVEFUNCTION_IMPL_H_ */
