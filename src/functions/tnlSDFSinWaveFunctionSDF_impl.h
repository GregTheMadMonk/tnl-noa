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

#ifndef TNLSDFSINWAVEFUNCTIONSDF_IMPL_H_
#define TNLSDFSINWAVEFUNCTIONSDF_IMPL_H_

#include <functions/tnlSDFSinWaveFunctionSDF.h>

   template< int dimensions, typename Real >
   tnlSDFSinWaveFunctionSDFBase< dimensions, Real >::tnlSDFSinWaveFunctionSDFBase()
   : waveLength( 1.0 ),
     amplitude( 1.0 ),
     phase( 0 ),
     wavesNumber( 0 )
   {
   }

   template< int dimensions, typename Real >
   bool tnlSDFSinWaveFunctionSDFBase< dimensions, Real >::setup( const tnlParameterContainer& parameters,
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

   template< int dimensions, typename Real >
   void tnlSDFSinWaveFunctionSDFBase< dimensions, Real >::setWaveLength( const Real& waveLength )
   {
      this->waveLength = waveLength;
   }

   template< int dimensions, typename Real >
   Real tnlSDFSinWaveFunctionSDFBase< dimensions, Real >::getWaveLength() const
   {
      return this->waveLength;
   }

   template< int dimensions, typename Real >
   void tnlSDFSinWaveFunctionSDFBase< dimensions, Real >::setAmplitude( const Real& amplitude )
   {
      this->amplitude = amplitude;
   }

   template< int dimensions, typename Real >
   Real tnlSDFSinWaveFunctionSDFBase< dimensions, Real >::getAmplitude() const
   {
      return this->amplitude;
   }

   template< int dimensions, typename Real >
   void tnlSDFSinWaveFunctionSDFBase< dimensions, Real >::setPhase( const Real& phase )
   {
      this->phase = phase;
   }

   template< int dimensions, typename Real >
   Real tnlSDFSinWaveFunctionSDFBase< dimensions, Real >::getPhase() const
   {
      return this->phase;
   }

   template< int dimensions, typename Real >
   void tnlSDFSinWaveFunctionSDFBase< dimensions, Real >::setWavesNumber( const Real& wavesNumber )
   {
      this->wavesNumber = wavesNumber;
   }

   template< int dimensions, typename Real >
   Real tnlSDFSinWaveFunctionSDFBase< dimensions, Real >::getWavesNumber() const
   {
      return this->wavesNumber;
   }


   template< typename Real >
      template< int XDiffOrder,
                int YDiffOrder,
                int ZDiffOrder>
   __cuda_callable__
   Real
   tnlSDFSinWaveFunctionSDF< 1, Real >::
   getPartialDerivative( const VertexType& v,
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
                int ZDiffOrder>
   __cuda_callable__
   Real
   tnlSDFSinWaveFunctionSDF< 2, Real >::
   getPartialDerivative( const VertexType& v,
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
                int ZDiffOrder>
   __cuda_callable__
   Real
   tnlSDFSinWaveFunctionSDF< 3, Real >::
   getPartialDerivative( const VertexType& v,
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
