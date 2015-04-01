/***************************************************************************
                          tnlSDFSinWaveFunctionSDF_impl.h  -  description
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

#ifndef TNLSDFSINWAVEFUNCTIONSDF_IMPL_H_
#define TNLSDFSINWAVEFUNCTIONSDF_IMPL_H_

#include <functions/tnlSDFSinWaveFunctionSDF.h>

template< typename Real >
tnlSDFSinWaveFunctionSDFBase< Real >::tnlSDFSinWaveFunctionSDFBase()
: waveLength( 0 ),
  amplitude( 0 ),
  phase( 0 ),
  wavesNumber( 0 )
{
}

template< typename Real >
bool tnlSDFSinWaveFunctionSDFBase< Real >::setup( const tnlParameterContainer& parameters,
        const tnlString& prefix )
{
   this->waveLength = parameters.getParameter< double >( prefix + "wave-length" );
   this->amplitude = parameters.getParameter< double >( prefix + "amplitude" );
   this->phase = parameters.getParameter< double >( prefix + "phase" );
   while(this->phase >2.0*M_PI)
	   this->phase -= 2.0*M_PI;
   this->wavesNumber = ceil( parameters.getParameter< double >( prefix + "waves-number" ) );
   return true;
}

template< typename Real >
void tnlSDFSinWaveFunctionSDFBase< Real >::setWaveLength( const Real& waveLength )
{
   this->waveLength = waveLength;
}

template< typename Real >
Real tnlSDFSinWaveFunctionSDFBase< Real >::getWaveLength() const
{
   return this->waveLength;
}

template< typename Real >
void tnlSDFSinWaveFunctionSDFBase< Real >::setAmplitude( const Real& amplitude )
{
   this->amplitude = amplitude;
}

template< typename Real >
Real tnlSDFSinWaveFunctionSDFBase< Real >::getAmplitude() const
{
   return this->amplitude;
}

template< typename Real >
void tnlSDFSinWaveFunctionSDFBase< Real >::setWavesNumber( const Real& wavesNumber )
{
   this->wavesNumber = wavesNumber;
}

template< typename Real >
Real tnlSDFSinWaveFunctionSDFBase< Real >::getWavesNumber() const
{
   return this->wavesNumber;
}

template< typename Real >
void tnlSDFSinWaveFunctionSDFBase< Real >::setPhase( const Real& phase )
{
   this->phase = phase;
}

template< typename Real >
Real tnlSDFSinWaveFunctionSDFBase< Real >::getPhase() const
{
   return this->phase;
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
Real tnlSDFSinWaveFunctionSDF< 1, Real >::getValue( const Vertex& v,
		const Real& time ) const
{
   const RealType& x = v.x();
   RealType distance;
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
	   distance = sqrt(x*x) + (this->phase)*(this->waveLength)/(2.0*M_PI);
   	   if (distance < (this->wavesNumber)*(this->waveLength)|| this->wavesNumber == 0.0)
	      return Sign(distance - round((2.0 * distance)/this->waveLength)* this->waveLength/2.0)
	    		    *(distance - round((2.0 * distance)/this->waveLength)* this->waveLength/2.0)
	    		    *Sign(sin(2.0 * M_PI * distance / this->waveLength));
   	   else
   		   return distance - (this->wavesNumber)*(this->waveLength);

   return 0.0;
}


template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
Real tnlSDFSinWaveFunctionSDF< 2, Real >::getValue( const Vertex& v,
		const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();

   if( XDiffOrder == 0 && YDiffOrder == 0)
   {
	   RealType distance  = sqrt(x*x + y*y) + (this->phase)*(this->waveLength)/(2.0*M_PI);
   	  if (distance < (this->wavesNumber)*(this->waveLength)|| this->wavesNumber == 0.0)
   			  return Sign(distance - round((2.0 *  distance)/this->waveLength)* this->waveLength/2.0)
   					    *(distance - round((2.0 * distance)/this->waveLength)* this->waveLength/2.0)
   					    *Sign(sin(2.0 * M_PI * distance / this->waveLength));
  	   else
  		   return distance - (this->wavesNumber)*(this->waveLength);
   }
   return 0.0;
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
Real tnlSDFSinWaveFunctionSDF< 3, Real >::getValue( const Vertex& v,
		const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   const RealType& z = v.z();
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
   {
	   RealType distance  = sqrt(x*x + y*y +z*z) + (this->phase)*(this->waveLength)/(2.0*M_PI);
   	   if (distance < (this->wavesNumber)*(this->waveLength)|| this->wavesNumber == 0.0)
	      return Sign(distance - round((2.0 * distance)/this->waveLength )* this->waveLength/2.0)
	    		    *(distance - round((2.0 * distance)/this->waveLength )* this->waveLength/2.0)
	    		    *Sign(sin( 2.0 * M_PI * distance / this->waveLength));
   	   else
   		   return distance - (this->wavesNumber)*(this->waveLength);
   }
   return 0.0;
}

#endif /* TNLSDFSINWAVEFUNCTIONSDF_IMPL_H_ */
