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

#pragma once

#include <functions/tnlSinWaveFunction.h>

template< int dimensions, typename Real >
tnlSinWaveFunctionBase< dimensions, Real >::tnlSinWaveFunctionBase()
: waveLength( 1.0 ),
  amplitude( 1.0 ),
  phase( 0 ),
  wavesNumber( 0 )
{
}

template< int dimensions, typename Real >
bool tnlSinWaveFunctionBase< dimensions, Real >::setup( const tnlParameterContainer& parameters,
                                           const tnlString& prefix )
{
   this->waveLength = parameters.getParameter< double >( prefix + "wave-length" );
   this->amplitude = parameters.getParameter< double >( prefix + "amplitude" );
   this->phase = parameters.getParameter< double >( prefix + "phase" );
   this->wavesNumber = parameters.getParameter< double >( prefix + "waves-number" );
   return true;
}

template< int dimensions, typename Real >
void tnlSinWaveFunctionBase< dimensions, Real >::setWaveLength( const Real& waveLength )
{
   this->waveLength = waveLength;
}

template< int dimensions, typename Real >
Real tnlSinWaveFunctionBase< dimensions, Real >::getWaveLength() const
{
   return this->waveLength;
}

template< int dimensions, typename Real >
void tnlSinWaveFunctionBase< dimensions, Real >::setAmplitude( const Real& amplitude )
{
   this->amplitude = amplitude;
}

template< int dimensions, typename Real >
Real tnlSinWaveFunctionBase< dimensions, Real >::getAmplitude() const
{
   return this->amplitude;
}

template< int dimensions, typename Real >
void tnlSinWaveFunctionBase< dimensions, Real >::setPhase( const Real& phase )
{
   this->phase = phase;
}

template< int dimensions, typename Real >
Real tnlSinWaveFunctionBase< dimensions, Real >::getPhase() const
{
   return this->phase;
}

template< int dimensions, typename Real >
void tnlSinWaveFunctionBase< dimensions, Real >::setWavesNumber( const Real& wavesNumber )
{
   this->wavesNumber = wavesNumber;
}

template< int dimensions, typename Real >
Real tnlSinWaveFunctionBase< dimensions, Real >::getWavesNumber() const
{
   return this->wavesNumber;
}

template< int dimensions, typename Real >
bool tnlSinWaveFunctionBase< dimensions, Real >::isInsideWaves( const Real& distance ) const
{
   if( this->wavesNumber == 0.0 ||
      distance + ( this->phase ) * ( this->waveLength ) / ( 2.0*M_PI ) < ( this->wavesNumber ) * (this->waveLength) )
      return true;
   return false;
   
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__
Real
tnlSinWaveFunction< 1, Real >::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;

   const RealType& x = v.x();
   const RealType distance = sqrt( x * x );
   
   if( XDiffOrder == 0 )
   {
      if( this->isInsideWaves( distance ) )
         return this->amplitude * sin( this->phase + 2.0 * M_PI * distance / this->waveLength );
	   else
	      return distance + this->phase * this->waveLength / ( 2.0*M_PI ) - this->wavesNumber * this->waveLength;
      
      /*RealType arg = 2.0 * M_PI * x  / this->waveLength;
      if( this->wavesNumber )
      {
         if( tnlAbs( arg ) > this->wavesNumber )
            arg = Sign( x ) * this->wavesNumber;
      }
      //cout << "arg = " << arg << " amplitude = " << this->amplitude << " -> " << this->amplitude * sin( this->phase + arg ) << endl;
      return this->amplitude * sin( this->phase + arg );*/
   }
   if( XDiffOrder == 1 )
   {
      if( this->isInsideWaves( distance ) )
         return 2.0 * M_PI / this->waveLength * this->amplitude * cos( this->phase + 2.0 * M_PI * sqrt(x*x) / this->waveLength );
	   else return  x / distance;
   }
   if( XDiffOrder == 2 )
   {
      if( this->isInsideWaves( distance ) )
         return -4.0 * M_PI * M_PI / ( this->waveLength * this->waveLength ) * this->amplitude * sin( this->phase + 2.0 * M_PI * x / this->waveLength );      
      else
      {
         tnlAssert( false, );
      }
   }
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
tnlSinWaveFunction< 1, Real >::
operator()( const VertexType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}



template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__
Real
tnlSinWaveFunction< 2, Real >::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   if ( ZDiffOrder != 0 )
      return 0.0;

   const RealType& x = v.x();
   const RealType& y = v.y();
   const RealType distance = sqrt( x * x + y * y);

   if( XDiffOrder == 0 && YDiffOrder == 0)
   {
      if( this->isInsideWaves( distance ) )
         return this->amplitude * sin( this->phase + 2.0 * M_PI * distance / this->waveLength );
	   else
	      return distance + this->phase * this->waveLength / ( 2.0*M_PI ) - this->wavesNumber * this->waveLength;
   }
   
   if( XDiffOrder == 1 && YDiffOrder == 0 )
   {
	   if( this->isInsideWaves( distance ) )
		   return 2.0 * M_PI / this->waveLength * this->amplitude * cos( this->phase * 2.0 * M_PI * distance / this->waveLength ) * x / distance;
      return  x / distance;
   }
   if( XDiffOrder == 0 && YDiffOrder == 1 )
   {
	   if( this->isInsideWaves( distance ) )
		   return 2.0 * M_PI / this->waveLength * this->amplitude * cos( this->phase * 2.0 * M_PI * distance / this->waveLength ) * y / distance;
	   return y / distance;
   }		
   if( XDiffOrder == 1 && YDiffOrder == 0 )
   {
  	   if( this->isInsideWaves( distance ) )
         return 2.0 * M_PI * x / ( this->waveLength * distance ) * this->amplitude * cos( this->phase + 2.0 * M_PI * distance / this->waveLength );
      tnlAssert( false, "TODO: implement this" );
   }
   if( XDiffOrder == 2 && YDiffOrder == 0 )
   {
  	   if( this->isInsideWaves( distance ) )
         return 2.0 * M_PI * x * x / ( this->waveLength * distance * distance * distance ) * this->amplitude * cos( this->phase + 2.0 * M_PI * distance / this->waveLength ) - 4.0 * M_PI * M_PI * x * x / ( this->waveLength * this->waveLength * ( x * x + y * y ) ) * this->amplitude * sin( this->phase + 2.0 * M_PI * distance / this->waveLength );
      tnlAssert( false, "TODO: implement this" );
   }
   if( XDiffOrder == 0 && YDiffOrder == 1 )
   {
  	   if( this->isInsideWaves( distance ) )
         return 2.0 * M_PI * y / ( this->waveLength * distance ) * this->amplitude * cos( this->phase + 2.0 * M_PI * distance / this->waveLength );
      tnlAssert( false, "TODO: implement this" );
   }
   if( XDiffOrder == 0 && YDiffOrder == 2 )
   {
  	   if( this->isInsideWaves( distance ) )
         return 2.0 * M_PI * y * y / ( this->waveLength * distance * distance * distance ) * this->amplitude * cos( this->phase + 2.0 * M_PI * distance / this->waveLength ) - 4.0 * M_PI * M_PI * y * y / ( this->waveLength * this->waveLength * ( x * x + y * y ) ) * this->amplitude * sin( this->phase + 2.0 * M_PI * distance / this->waveLength );
      tnlAssert( false, "TODO: implement this" );
   }
   if( XDiffOrder == 1 && YDiffOrder == 1 )
   {
  	   if( this->isInsideWaves( distance ) )
         return -4.0 * M_PI * M_PI * x * y / ( this->waveLength * this->waveLength * (x * x + y * y ) )* this->amplitude * sin( this->phase + 2.0 * M_PI * distance / this->waveLength ) 
             - 2.0 * M_PI * this->amplitude * x * y * cos( this->phase + 2.0 * M_PI * distance / this->waveLength ) / ( this->waveLength  * sqrt( (x * x + y * y )  * (x * x + y * y ) * (x * x + y * y ) ) );
      tnlAssert( false, "TODO: implement this" );
   }
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
tnlSinWaveFunction< 2, Real >::
operator()( const VertexType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}


template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__
Real
tnlSinWaveFunction< 3, Real >::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   const RealType& z = v.z();
   const RealType distance = sqrt( x * x + y * y + z * z );
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
   {
      if( this->isInsideWaves( distance ) )
         return this->amplitude * sin( this->phase + 2.0 * M_PI * distance / this->waveLength );
   }
   if( XDiffOrder == 1 && YDiffOrder == 0 && ZDiffOrder == 0 )
   {
      if( this->isInsideWaves( distance ) )
         return 2.0 * M_PI * x / ( this->waveLength * distance ) * this->amplitude * cos( this->phase + 2.0 * M_PI * distance / this->waveLength );
		return x / distance;
   }
   if( XDiffOrder == 2 && YDiffOrder == 0 && ZDiffOrder == 0 )
   {
      if( this->isInsideWaves( distance ) )
         return 2.0 * M_PI * ( y * y + z * z ) / ( this->waveLength * distance * distance * distance ) * this->amplitude * cos( this->phase + 2.0 * M_PI * distance / this->waveLength ) - 4.0 * M_PI * M_PI * x * x / ( this->waveLength * this->waveLength * ( x * x + y * y + z * z ) ) * this->amplitude * sin( this->phase + 2.0 * M_PI * distance / this->waveLength );
      tnlAssert( false, "TODO: implement this" );
   }
   if( XDiffOrder == 0 && YDiffOrder == 1 && ZDiffOrder == 0 )
   {
      if( this->isInsideWaves( distance ) )
         return 2.0 * M_PI * y / ( this->waveLength * distance ) * this->amplitude * cos( this->phase + 2.0 * M_PI * distance / this->waveLength );
		return y / distance;         
   }
   if( XDiffOrder == 0 && YDiffOrder == 2 && ZDiffOrder == 0 )
   {
      if( this->isInsideWaves( distance ) )
         return 2.0 * M_PI * ( x * x + z * z ) / ( this->waveLength * distance * distance * distance ) * this->amplitude * cos( this->phase + 2.0 * M_PI * distance / this->waveLength ) - 4.0 * M_PI * M_PI * y * y / ( this->waveLength * this->waveLength * ( x * x + y * y + z * z ) ) * this->amplitude * sin( this->phase + 2.0 * M_PI * distance / this->waveLength );
      tnlAssert( false, "TODO: implement this" );
   }
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 1 )
   {
      if( this->isInsideWaves( distance ) )
         return 2.0 * M_PI * z / ( this->waveLength * distance ) * this->amplitude * cos( this->phase + 2.0 * M_PI * distance / this->waveLength );
		return z / distance;
   }
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 2 )
   {
      if( this->isInsideWaves( distance ) )
         return 2.0 * M_PI * ( x * x + y * y ) / ( this->waveLength * distance * distance * distance ) * this->amplitude * cos( this->phase + 2.0 * M_PI * distance / this->waveLength ) - 4.0 * M_PI * M_PI * z * z / ( this->waveLength * this->waveLength * ( x * x + y * y + z * z ) ) * this->amplitude * sin( this->phase + 2.0 * M_PI * distance / this->waveLength ); 
      tnlAssert( false, "TODO: implement this" );
   }
   if( XDiffOrder == 1 && YDiffOrder == 1 && ZDiffOrder == 0 )
   {
      if( this->isInsideWaves( distance ) )
         return -4.0 * M_PI * M_PI * x * y / ( this->waveLength * this->waveLength * (x * x + y * y + z * z ) )* this->amplitude * sin( this->phase + 2.0 * M_PI * distance / this->waveLength ) 
             - 2.0 * M_PI * this->amplitude * x * y * cos( this->phase + 2.0 * M_PI * distance / this->waveLength ) / ( this->waveLength  * sqrt( (x * x + y * y + z * z )  * (x * x + y * y + z * z ) * (x * x + y * y + z * z ) ) );
      tnlAssert( false, "TODO: implement this" );
   }
   if( XDiffOrder == 1 && YDiffOrder == 0 && ZDiffOrder == 1 )
   {
      if( this->isInsideWaves( distance ) )
         return -4.0 * M_PI * M_PI * x * z / ( this->waveLength * this->waveLength * (x * x + y * y + z * z ) )* this->amplitude * sin( this->phase + 2.0 * M_PI * distance / this->waveLength ) 
                - 2.0 * M_PI * this->amplitude * x * z * cos( this->phase + 2.0 * M_PI * distance / this->waveLength ) / ( this->waveLength  * sqrt( (x * x + y * y + z * z )  * (x * x + y * y + z * z ) * (x * x + y * y + z * z ) ) );
      tnlAssert( false, "TODO: implement this" );
   }
   if( XDiffOrder == 0 && YDiffOrder == 1 && ZDiffOrder == 1 )
   {
      if( this->isInsideWaves( distance ) )
         return -4.0 * M_PI * M_PI * z * y / ( this->waveLength * this->waveLength * (x * x + y * y + z * z ) )* this->amplitude * sin( this->phase + 2.0 * M_PI * distance / this->waveLength ) 
                - 2.0 * M_PI * this->amplitude * z * y * cos( this->phase + 2.0 * M_PI * distance / this->waveLength ) / ( this->waveLength  * sqrt( (x * x + y * y + z * z )  * (x * x + y * y + z * z ) * (x * x + y * y + z * z ) ) );
      tnlAssert( false, "TODO: implement this" );
   }
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
tnlSinWaveFunction< 3, Real >::
operator()( const VertexType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}

