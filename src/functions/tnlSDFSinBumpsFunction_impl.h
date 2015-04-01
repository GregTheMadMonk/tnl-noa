/***************************************************************************
                          tnlSDFSinBumpsFunction_impl.h  -  description
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

#ifndef TNLSDFSINBUMPSFUNCTION_IMPL_H_
#define TNLSDFSINBUMPSFUNCTION_IMPL_H_

#include <functions/tnlSDFSinBumpsFunction.h>

template< typename Vertex >
void tnlSDFSinBumpsFunctionBase< Vertex >::setWaveLength( const Vertex& waveLength )
{
   this->waveLength = waveLength;
}

template< typename Vertex >
const Vertex& tnlSDFSinBumpsFunctionBase< Vertex >::getWaveLength() const
{
   return this->waveLength;
}

template< typename Vertex >
void tnlSDFSinBumpsFunctionBase< Vertex >::setWavesNumber( const Vertex& waveLength )
{
   this->wavesNumber = wavesNumber;
}

template< typename Vertex >
const Vertex& tnlSDFSinBumpsFunctionBase< Vertex >::getWavesNumber() const
{
   return this->wavesNumber;
}

template< typename Vertex >
void tnlSDFSinBumpsFunctionBase< Vertex >::setAmplitude( const typename Vertex::RealType& amplitude )
{
   this->amplitude = amplitude;
}

template< typename Vertex >
const typename Vertex::RealType& tnlSDFSinBumpsFunctionBase< Vertex >::getAmplitude() const
{
   return this->amplitude;
}

template< typename Vertex >
void tnlSDFSinBumpsFunctionBase< Vertex >::setPhase( const Vertex& phase )
{
   this->phase = phase;
}

template< typename Vertex >
const Vertex& tnlSDFSinBumpsFunctionBase< Vertex >::getPhase() const
{
   return this->phase;
}

/***
 * 1D
 */

template< typename Real >
tnlSDFSinBumpsFunction< 1, Real >::tnlSDFSinBumpsFunction()
{
}

template< typename Real >
bool tnlSDFSinBumpsFunction< 1, Real >::setup( const tnlParameterContainer& parameters,
        const tnlString& prefix)
{
   this->amplitude = parameters.getParameter< double >( prefix+"amplitude" );
   this->waveLength.x() = parameters.getParameter< double >( prefix+"wave-length-x" );
   while(this->waveLength.x() > 2.0*M_PI)
	   this->waveLength.x() -= 2.0*M_PI;
   this->wavesNumber.x() = ceil( parameters.getParameter< double >( prefix+"waves-number-x" ) );
   this->phase.x() = parameters.getParameter< double >( prefix+"phase-x" );
   return true;
}


template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
Real tnlSDFSinBumpsFunction< 1, Real >::getValue( const Vertex& v,
              const Real& time ) const
{
   const RealType& x = v.x();
   if (sqrt(x*x) + Sign(x)*(this->phase.x())*(this->waveLength.x())/(2.0*M_PI) > this->wavesNumber.x()*this->waveLength.x() && this->wavesNumber.x() != 0.0 )
	   return 0.0;
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
      return this->amplitude * sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() );
   if( XDiffOrder == 1 )
      return 2.0 * M_PI / this->waveLength.x() * this->amplitude * cos( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() );
   return 0.0;
}

/****
 * 2D
 */

template< typename Real >
tnlSDFSinBumpsFunction< 2, Real >::tnlSDFSinBumpsFunction()
{
}

template< typename Real >
bool tnlSDFSinBumpsFunction< 2, Real >::setup( const tnlParameterContainer& parameters,
        const tnlString& prefix )
{
   this->amplitude = parameters.getParameter< double >( prefix+"amplitude" );
   this->waveLength.x() = parameters.getParameter< double >( prefix+"wave-length-x" );
   this->waveLength.y() = parameters.getParameter< double >( prefix+"wave-length-y" );
   while(this->waveLength.x() > 2.0*M_PI)
	   this->waveLength.x() -= 2.0*M_PI;
   while(this->waveLength.y() > 2.0*M_PI)
	   this->waveLength.y() -= 2.0*M_PI;
   this->wavesNumber.x() = ceil( parameters.getParameter< double >( prefix+"waves-number-x" ) );
   this->wavesNumber.y() = ceil( parameters.getParameter< double >( prefix+"waves-number-y" ) );
   this->phase.x() = parameters.getParameter< double >( prefix+"phase-x" );
   this->phase.y() = parameters.getParameter< double >( prefix+"phase-y" );
   return true;
}


template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
Real tnlSDFSinBumpsFunction< 2, Real >::getValue( const Vertex& v,
              const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   if ( (sqrt(x*x) + Sign(x)*(this->phase.x())*(this->waveLength.x())/(2.0*M_PI) > this->wavesNumber.x()*this->waveLength.x() && this->wavesNumber.x() != 0.0 )  ||
	    (sqrt(y*y) + Sign(y)*(this->phase.y())*(this->waveLength.y())/(2.0*M_PI) > this->wavesNumber.y()*this->waveLength.y() && this->wavesNumber.y() != 0.0 ) )
	   return 0.0;
   if( ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 && YDiffOrder == 0 )
      return this->amplitude *
             sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) *
             sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() );
   else if( XDiffOrder == 1 && YDiffOrder == 0 )
	      return this->amplitude *
	             cos( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) *
	             sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) *
	             2.0*M_PI/this->waveLength.x();
   else if( XDiffOrder == 0 && YDiffOrder == 1 )
	      return this->amplitude *
	             sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) *
	             cos( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) *
	             2.0*M_PI/this->waveLength.y();

   return 0.0;
}

/****
 * 3D
 */

template< typename Real >
tnlSDFSinBumpsFunction< 3, Real >::tnlSDFSinBumpsFunction()
{
}

template< typename Real >
bool tnlSDFSinBumpsFunction< 3, Real >::setup( const tnlParameterContainer& parameters,
        const tnlString& prefix )
{
   this->amplitude = parameters.getParameter< double >( prefix+"amplitude" );
   this->waveLength.x() = parameters.getParameter< double >( prefix+"wave-length-x" );
   this->waveLength.y() = parameters.getParameter< double >( prefix+"wave-length-y" );
   this->waveLength.z() = parameters.getParameter< double >( prefix+"wave-length-z" );
   while(this->waveLength.x() > 2.0*M_PI)
	   this->waveLength.x() -= 2.0*M_PI;
   while(this->waveLength.y() > 2.0*M_PI)
	   this->waveLength.y() -= 2.0*M_PI;
   while(this->waveLength.z() > 2.0*M_PI)
	   this->waveLength.z() -= 2.0*M_PI;
   this->wavesNumber.x() = ceil( parameters.getParameter< double >( prefix+"waves-number-x" ) );
   this->wavesNumber.y() = ceil( parameters.getParameter< double >( prefix+"waves-number-y" ) );
   this->wavesNumber.z() = ceil( parameters.getParameter< double >( prefix+"waves-number-z" ) );
   this->phase.x() = parameters.getParameter< double >( prefix+"phase-x" );
   this->phase.y() = parameters.getParameter< double >( prefix+"phase-y" );
   this->phase.z() = parameters.getParameter< double >(prefix+"phase-z" );
   return true;
}


template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
Real tnlSDFSinBumpsFunction< 3, Real >::getValue( const Vertex& v,
              const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   const RealType& z = v.z();
   if ( (sqrt(x*x) + Sign(x)*(this->phase.x())*(this->waveLength.x())/(2.0*M_PI) > this->wavesNumber.x()*this->waveLength.x() && this->wavesNumber.x() != 0.0 ) ||
		(sqrt(y*y) + Sign(y)*(this->phase.y())*(this->waveLength.y())/(2.0*M_PI) > this->wavesNumber.y()*this->waveLength.y() && this->wavesNumber.y() != 0.0 ) ||
		(sqrt(z*z) + Sign(z)*(this->phase.z())*(this->waveLength.z())/(2.0*M_PI) > this->wavesNumber.z()*this->waveLength.z() && this->wavesNumber.z() != 0.0 ) )
	   return 0.0;

   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
      return this->amplitude *
             sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) *
             sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) *
             sin( this->phase.z() + 2.0 * M_PI * z / this->waveLength.z() );
   else if( XDiffOrder == 1 && YDiffOrder == 0 && ZDiffOrder == 0 )
	      return this->amplitude *
	             cos( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) *
	             sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) *
	             sin( this->phase.z() + 2.0 * M_PI * z / this->waveLength.z() ) *
	             2.0*M_PI/this->waveLength.x();
   else if( XDiffOrder == 0 && YDiffOrder == 1 && ZDiffOrder == 0 )
	      return this->amplitude *
	             sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) *
	             cos( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) *
	             sin( this->phase.z() + 2.0 * M_PI * z / this->waveLength.z() ) *
	             2.0*M_PI/this->waveLength.y();
   else if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 1 )
	      return this->amplitude *
	             sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) *
	             sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) *
	             cos( this->phase.z() + 2.0 * M_PI * z / this->waveLength.z() ) *
	             2.0*M_PI/this->waveLength.z();
   return 0.0;
}

#endif /* TNLSDFSINBUMPSFUNCTION_IMPL_H_ */
