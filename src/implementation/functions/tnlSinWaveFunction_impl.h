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

#include <functions/tnlSinWaveFunction.h>

template< typename Real >
tnlSinWaveFunctionBase< Real >::tnlSinWaveFunctionBase()
: waveLength( 1.0 ),
  amplitude( 1.0 ),
  phase( 0 ),
  wavesNumber( 0 )
{
}

template< typename Real >
bool tnlSinWaveFunctionBase< Real >::setup( const tnlParameterContainer& parameters,
                                           const tnlString& prefix )
{
   this->waveLength = parameters.GetParameter< double >( prefix + "wave-length" );
   this->amplitude = parameters.GetParameter< double >( prefix + "amplitude" );
   this->phase = parameters.GetParameter< double >( prefix + "phase" );
   parameters.GetParameter< double >( prefix + "waves-number" );
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

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
Real
tnlSinWaveFunction< 1, Real >::
getValue( const Vertex& v,
          const Real& time ) const
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
      //cout << "arg = " << arg << " amplitude = " << this->amplitude << " -> " << this->amplitude * sin( this->phase + arg ) << endl;
      return this->amplitude * sin( this->phase + arg );
   }
   if( XDiffOrder == 1 )
      return 2.0 * M_PI / this->waveLength * this->amplitude * cos( this->phase + 2.0 * M_PI * x / this->waveLength );
   if( XDiffOrder == 2 )
      return -4.0 * M_PI * M_PI / ( this->waveLength * this->waveLength ) * this->amplitude * sin( this->phase + 2.0 * M_PI * x / this->waveLength );
   return 0.0;
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
tnlSinWaveFunction< 2, Real >::
getValue( const Vertex& v,
          const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   if ( ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 && YDiffOrder == 0)
   {
      return this->amplitude * sin( this->phase + 2.0 * M_PI * sqrt( x * x + y * y ) / this->waveLength );
   }
   if( XDiffOrder == 1 && YDiffOrder == 0 )
      return 2.0 * M_PI * x / ( this->waveLength * sqrt( x * x + y * y ) ) * this->amplitude * cos( this->phase + 2.0 * M_PI * sqrt( x * x + y * y ) / this->waveLength );
   if( XDiffOrder == 2 && YDiffOrder == 0 )
      return 2.0 * M_PI * x * x / ( this->waveLength * sqrt( x * x + y * y ) * sqrt( x * x + y * y ) * sqrt( x * x + y * y ) ) * this->amplitude * cos( this->phase + 2.0 * M_PI * sqrt( x * x + y * y ) / this->waveLength ) - 4.0 * M_PI * M_PI * x * x / ( this->waveLength * this->waveLength * ( x * x + y * y ) ) * this->amplitude * sin( this->phase + 2.0 * M_PI * sqrt( x * x + y * y ) / this->waveLength );
   if( XDiffOrder == 0 && YDiffOrder == 1 )
      return 2.0 * M_PI * y / ( this->waveLength * sqrt( x * x + y * y ) ) * this->amplitude * cos( this->phase + 2.0 * M_PI * sqrt( x * x + y * y ) / this->waveLength );
   if( XDiffOrder == 0 && YDiffOrder == 2 )
      return 2.0 * M_PI * y * y / ( this->waveLength * sqrt( x * x + y * y ) * sqrt( x * x + y * y ) * sqrt( x * x + y * y ) ) * this->amplitude * cos( this->phase + 2.0 * M_PI * sqrt( x * x + y * y ) / this->waveLength ) - 4.0 * M_PI * M_PI * y * y / ( this->waveLength * this->waveLength * ( x * x + y * y ) ) * this->amplitude * sin( this->phase + 2.0 * M_PI * sqrt( x * x + y * y ) / this->waveLength );
   return 0.0;
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
tnlSinWaveFunction< 3, Real >::
getValue( const Vertex& v,
          const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   const RealType& z = v.z();
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
   {
      return this->amplitude * sin( this->phase + 2.0 * M_PI * sqrt( x * x + y * y + z * z ) / this->waveLength );
   }
   if( XDiffOrder == 1 && YDiffOrder == 0 && ZDiffOrder == 0 )
      return 2.0 * M_PI * x / ( this->waveLength * sqrt( x * x + y * y + z * z ) ) * this->amplitude * cos( this->phase + 2.0 * M_PI * sqrt( x * x + y * y + z * z ) / this->waveLength );
   if( XDiffOrder == 2 && YDiffOrder == 0 && ZDiffOrder == 0 )
      return 2.0 * M_PI * ( y * y + z * z ) / ( this->waveLength * sqrt( x * x + y * y + z * z ) * sqrt( x * x + y * y + z * z ) * sqrt( x * x + y * y + z * z ) ) * this->amplitude * cos( this->phase + 2.0 * M_PI * sqrt( x * x + y * y + z * z ) / this->waveLength ) - 4.0 * M_PI * M_PI * x * x / ( this->waveLength * this->waveLength * ( x * x + y * y + z * z ) ) * this->amplitude * sin( this->phase + 2.0 * M_PI * sqrt( x * x + y * y + z * z ) / this->waveLength );
   if( XDiffOrder == 0 && YDiffOrder == 1 && ZDiffOrder == 0 )
      return 2.0 * M_PI * y / ( this->waveLength * sqrt( x * x + y * y + z * z ) ) * this->amplitude * cos( this->phase + 2.0 * M_PI * sqrt( x * x + y * y + z * z ) / this->waveLength );
   if( XDiffOrder == 0 && YDiffOrder == 2 && ZDiffOrder == 0 )
      return 2.0 * M_PI * ( x * x + z * z ) / ( this->waveLength * sqrt( x * x + y * y + z * z ) * sqrt( x * x + y * y + z * z ) * sqrt( x * x + y * y + z * z ) ) * this->amplitude * cos( this->phase + 2.0 * M_PI * sqrt( x * x + y * y + z * z ) / this->waveLength ) - 4.0 * M_PI * M_PI * y * y / ( this->waveLength * this->waveLength * ( x * x + y * y + z * z ) ) * this->amplitude * sin( this->phase + 2.0 * M_PI * sqrt( x * x + y * y + z * z ) / this->waveLength );
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 1 )
      return 2.0 * M_PI * z / ( this->waveLength * sqrt( x * x + y * y + z * z ) ) * this->amplitude * cos( this->phase + 2.0 * M_PI * sqrt( x * x + y * y + z * z ) / this->waveLength );
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 2 )
      return 2.0 * M_PI * ( x * x + y * y ) / ( this->waveLength * sqrt( x * x + y * y + z * z ) * sqrt( x * x + y * y + z * z ) * sqrt( x * x + y * y + z * z ) ) * this->amplitude * cos( this->phase + 2.0 * M_PI * sqrt( x * x + y * y + z * z ) / this->waveLength ) - 4.0 * M_PI * M_PI * z * z / ( this->waveLength * this->waveLength * ( x * x + y * y + z * z ) ) * this->amplitude * sin( this->phase + 2.0 * M_PI * sqrt( x * x + y * y + z * z ) / this->waveLength ); 
   return 0.0;
}

#endif /* TNLSINWAVEFUNCTION_IMPL_H_ */
