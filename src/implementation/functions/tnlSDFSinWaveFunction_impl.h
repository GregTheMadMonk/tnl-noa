/***************************************************************************
                          tnlSDFSinWaveFunction_impl.h  -  description
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

#ifndef TNLSDFSINWAVEFUNCTION_IMPL_H_
#define TNLSDFSINWAVEFUNCTION_IMPL_H_

#include <functions/tnlSDFSinWaveFunction.h>

template< typename Real >
tnlSDFSinWaveFunctionBase< Real >::tnlSDFSinWaveFunctionBase()
: waveLength( 0.0 ),
  amplitude( 1.0 ),
  phase( 0.0 ),
  wavesNumber( 0 )
{
}

template< typename Real >
bool tnlSDFSinWaveFunctionBase< Real >::setup( const tnlParameterContainer& parameters,
        const tnlString& prefix )
{
   this->waveLength = parameters.GetParameter< double >( prefix + "wave-length" );
   this->amplitude = parameters.GetParameter< double >( prefix + "amplitude" );
   this->phase = parameters.GetParameter< double >( prefix + "phase" );
   while(this->phase >2.0*M_PI)
	   this->phase -= 2.0*M_PI;
   this->wavesNumber = ceil( parameters.GetParameter< double >( prefix + "waves-number" ) );
   return true;
}

template< typename Real >
void tnlSDFSinWaveFunctionBase< Real >::setWaveLength( const Real& waveLength )
{
   this->waveLength = waveLength;
}

template< typename Real >
Real tnlSDFSinWaveFunctionBase< Real >::getWaveLength() const
{
   return this->waveLength;
}

template< typename Real >
void tnlSDFSinWaveFunctionBase< Real >::setAmplitude( const Real& amplitude )
{
   this->amplitude = amplitude;
}

template< typename Real >
Real tnlSDFSinWaveFunctionBase< Real >::getAmplitude() const
{
   return this->amplitude;
}

template< typename Real >
void tnlSDFSinWaveFunctionBase< Real >::setWavesNumber( const Real& wavesNumber )
{
   this->wavesNumber = wavesNumber;
}

template< typename Real >
Real tnlSDFSinWaveFunctionBase< Real >::getWavesNumber() const
{
   return this->wavesNumber;
}

template< typename Real >
void tnlSDFSinWaveFunctionBase< Real >::setPhase( const Real& phase )
{
   this->phase = phase;
}

template< typename Real >
Real tnlSDFSinWaveFunctionBase< Real >::getPhase() const
{
   return this->phase;
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
Real tnlSDFSinWaveFunction< 1, Real >::getValue( const Vertex& v,
		const Real& time ) const
{
   const RealType& x = v.x();
   RealType distance = sqrt(x*x);;
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
   {
	   if (distance + (this->phase)*(this->waveLength)/(2.0*M_PI) < (this->wavesNumber)*(this->waveLength) || this->wavesNumber == 0.0)
	      	return this->amplitude * sin( this->phase + 2.0 * M_PI * distance / this->waveLength );
	  else
	      	return distance  + (this->phase)*(this->waveLength)/(2.0*M_PI) - (this->wavesNumber)*(this->waveLength);
   }
   if( XDiffOrder == 1 )
   {
	   if (distance + (this->phase)*(this->waveLength)/(2.0*M_PI) < (this->wavesNumber)*(this->waveLength) || this->wavesNumber == 0.0)
		   return 2.0 * M_PI / this->waveLength * this->amplitude * cos( this->phase + 2.0 * M_PI * sqrt(x*x) / this->waveLength );
	   else return x/sqrt(x*x);
   }
   return 0.0;
}


template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
Real tnlSDFSinWaveFunction< 2, Real >::getValue( const Vertex& v,
		const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   RealType distance;
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
   {
	   RealType distance  = sqrt(x*x + y*y);
	   if (distance + (this->phase)*(this->waveLength)/(2.0*M_PI) < (this->wavesNumber)*(this->waveLength) || this->wavesNumber == 0.0)
   			   return this->amplitude * sin( this->phase + 2.0 * M_PI * distance / this->waveLength );
   	   else
   		   return distance  + (this->phase)*(this->waveLength)/(2.0*M_PI) - (this->wavesNumber)*(this->waveLength);
   }
   else if( XDiffOrder == 1 && YDiffOrder == 0 && ZDiffOrder == 0 )
   {
	   if (distance + (this->phase)*(this->waveLength)/(2.0*M_PI) < (this->wavesNumber)*(this->waveLength) || this->wavesNumber == 0.0)
		   return 2.0 * M_PI / this->waveLength * this->amplitude * cos( this->phase * 2.0 * M_PI * sqrt( x * x + y * y ) / this->waveLength )*x/sqrt( x * x + y * y );
	   else
		   return x/sqrt(x*x+y*y);
   }
   else if( XDiffOrder == 0 && YDiffOrder == 1 && ZDiffOrder == 0 )
   {
	   if (distance + (this->phase)*(this->waveLength)/(2.0*M_PI) < (this->wavesNumber)*(this->waveLength) || this->wavesNumber == 0.0)
		   return 2.0 * M_PI / this->waveLength * this->amplitude * cos( this->phase * 2.0 * M_PI * sqrt( x * x + y * y ) / this->waveLength )*y/sqrt( x * x + y * y );
	   else
		   return y/sqrt( x * x + y * y );
   }
   return 0.0;
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
Real tnlSDFSinWaveFunction< 3, Real >::getValue( const Vertex& v,
		const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   const RealType& z = v.z();
   RealType distance;
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
   {
	   RealType distance  = sqrt(x*x + y*y + z*z);
	   if (distance + (this->phase)*(this->waveLength)/(2.0*M_PI) < (this->wavesNumber)*(this->waveLength) || this->wavesNumber == 0.0)
   			   return this->amplitude * sin( this->phase + 2.0 * M_PI * sqrt( x * x + y * y + z * z ) / this->waveLength );
   	   else
   		   return distance  + (this->phase)*(this->waveLength)/(2.0*M_PI) - (this->wavesNumber)*(this->waveLength);
   }
   else if( XDiffOrder == 1 && YDiffOrder == 0 && ZDiffOrder == 0 )
   {
	   if (distance + (this->phase)*(this->waveLength)/(2.0*M_PI) < (this->wavesNumber)*(this->waveLength) || this->wavesNumber == 0.0)
		   return 2.0 * M_PI / this->waveLength * this->amplitude * cos( this->phase * 2.0 * M_PI * sqrt( x * x + y * y + z * z) / this->waveLength )*x/sqrt( x * x + y * y + z * z);
	   else
		   return x/sqrt( x * x + y * y + z * z);
   }
   else if( XDiffOrder == 0 && YDiffOrder == 1 && ZDiffOrder == 0 )
   {
	   if (distance + (this->phase)*(this->waveLength)/(2.0*M_PI) < (this->wavesNumber)*(this->waveLength) || this->wavesNumber == 0.0)
		   return 2.0 * M_PI / this->waveLength * this->amplitude * cos( this->phase * 2.0 * M_PI * sqrt( x * x + y * y + z * z) / this->waveLength )*y/sqrt( x * x + y * y + z * z);
	   else
		   return y/sqrt( x * x + y * y + z * z);
   }
   else if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 1 )
   {
	   if (distance + (this->phase)*(this->waveLength)/(2.0*M_PI) < (this->wavesNumber)*(this->waveLength) || this->wavesNumber == 0.0)
		   return 2.0 * M_PI / this->waveLength * this->amplitude * cos( this->phase * 2.0 * M_PI * sqrt( x * x + y * y + z * z) / this->waveLength )*z/sqrt( x * x + y * y + z * z);
	   else
		   return z/sqrt( x * x + y * y + z * z);
   }
   return 0.0;
}

#endif /* TNLSDFSINWAVEFUNCTION_IMPL_H_ */
