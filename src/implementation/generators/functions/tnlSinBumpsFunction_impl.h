/***************************************************************************
                          tnlSinBumpsFunction_impl.h  -  description
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

#ifndef TNLSINBUMPSFUNCTION_IMPL_H_
#define TNLSINBUMPSFUNCTION_IMPL_H_

#include <generators/functions/tnlSinBumpsFunction.h>

template< int Dimensions, typename Vertex, typename Device >
void tnlSinBumpsFunctionBase< Dimensions, Vertex, Device >::setWaveLength( const Vertex& waveLength )
{
   this->waveLength = waveLength;
}

template< int Dimensions, typename Vertex, typename Device >
const Vertex& tnlSinBumpsFunctionBase< Dimensions, Vertex, Device >::getWaveLength() const
{
   return this->waveLength;
}

template< int Dimensions, typename Vertex, typename Device >
void tnlSinBumpsFunctionBase< Dimensions, Vertex, Device >::setAmplitude( const typename Vertex::RealType& amplitude )
{
   this->amplitude = amplitude;
}

template< int Dimensions, typename Vertex, typename Device >
const typename Vertex::RealType& tnlSinBumpsFunctionBase< Dimensions, Vertex, Device >::getAmplitude() const
{
   return this->amplitude;
}

template< int Dimensions, typename Vertex, typename Device >
void tnlSinBumpsFunctionBase< Dimensions, Vertex, Device >::setPhase( const Vertex& phase )
{
   this->phase = phase;
}

template< int Dimensions, typename Vertex, typename Device >
const Vertex& tnlSinBumpsFunctionBase< Dimensions, Vertex, Device >::getPhase() const
{
   return this->phase;
}

/***
 * 1D
 */

template< typename Vertex, typename Device >
tnlSinBumpsFunction< 1, Vertex, Device >::tnlSinBumpsFunction()
{
}

template< typename Vertex, typename Device >
bool tnlSinBumpsFunction< 1, Vertex, Device >::init( const tnlParameterContainer& parameters )
{
   this->amplitude = parameters.GetParameter< double >( "amplitude" );
   this->waveLength.x() = parameters.GetParameter< double >( "wave-length-x" );
   this->phase.x() = parameters.GetParameter< double >( "phase-x" );
   return true;
}


template< typename Vertex, typename Device >
   template< int XDiffOrder, int YDiffOrder, int ZDiffOrder >
      typename Vertex::RealType tnlSinBumpsFunction< 1, Vertex, Device >::getF( const Vertex& v ) const
{
   const RealType& x = v.x();
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
      return this->amplitude * sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() );
   if( XDiffOrder == 1 )
      return 2.0 * M_PI / this->waveLength.x() * this->amplitude * cos( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() );
   if( XDiffOrder == 2 )
      return -4.0 * M_PI * M_PI / ( this->waveLength.x() * this->waveLength.x() ) * this->amplitude * sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() );
   return 0.0;
}

/****
 * 2D
 */

template< typename Vertex, typename Device >
tnlSinBumpsFunction< 2, Vertex, Device >::tnlSinBumpsFunction()
{
}

template< typename Vertex, typename Device >
bool tnlSinBumpsFunction< 2, Vertex, Device >::init( const tnlParameterContainer& parameters )
{
   this->amplitude = parameters.GetParameter< double >( "amplitude" );
   this->waveLength.x() = parameters.GetParameter< double >( "wave-length-x" );
   this->waveLength.y() = parameters.GetParameter< double >( "wave-length-y" );
   this->phase.x() = parameters.GetParameter< double >( "phase-x" );
   this->phase.y() = parameters.GetParameter< double >( "phase-y" );
   return true;
}


template< typename Vertex, typename Device >
   template< int XDiffOrder, int YDiffOrder, int ZDiffOrder >
      typename Vertex::RealType tnlSinBumpsFunction< 2, Vertex, Device >::getF( const Vertex& v ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   if( ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 && YDiffOrder == 0 )
      return this->amplitude *
             sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) *
             sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() );
   if( XDiffOrder == 1 && YDiffOrder == 0 )
      return 2.0 * M_PI / this->waveLength.x() * this->amplitude * cos( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) * sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() );
   if( XDiffOrder == 2 && YDiffOrder == 0 )
      return -4.0 * M_PI * M_PI / ( this->waveLength.x() * this->waveLength.x() ) * this->amplitude * sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) * sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() );
   if( XDiffOrder == 0 && YDiffOrder == 1 )
      return 2.0 * M_PI / this->waveLength.y() * this->amplitude * cos( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) * sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() );
   if( XDiffOrder == 0 && YDiffOrder == 2 )
      return -4.0 * M_PI * M_PI / ( this->waveLength.y() * this->waveLength.y() ) * this->amplitude * sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) * sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() );
   return 0.0;
}

/****
 * 3D
 */

template< typename Vertex, typename Device >
tnlSinBumpsFunction< 3, Vertex, Device >::tnlSinBumpsFunction()
{
}

template< typename Vertex, typename Device >
bool tnlSinBumpsFunction< 3, Vertex, Device >::init( const tnlParameterContainer& parameters )
{
   this->amplitude = parameters.GetParameter< double >( "amplitude" );
   this->waveLength.x() = parameters.GetParameter< double >( "wave-length-x" );
   this->waveLength.y() = parameters.GetParameter< double >( "wave-length-y" );
   this->waveLength.z() = parameters.GetParameter< double >( "wave-length-z" );
   this->phase.x() = parameters.GetParameter< double >( "phase-x" );
   this->phase.y() = parameters.GetParameter< double >( "phase-y" );
   this->phase.z() = parameters.GetParameter< double >( "phase-z" );
   return true;
}


template< typename Vertex, typename Device >
   template< int XDiffOrder, int YDiffOrder, int ZDiffOrder >
      typename Vertex::RealType tnlSinBumpsFunction< 3, Vertex, Device >::getF( const Vertex& v ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   const RealType& z = v.z();
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0)
      return this->amplitude *
             sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) *
             sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) *
             sin( this->phase.z() + 2.0 * M_PI * z / this->waveLength.z() );
   if( XDiffOrder == 1 && YDiffOrder == 0 && ZDiffOrder == 0)
      return 2.0 * M_PI / this->waveLength.x() * this->amplitude * cos( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) * sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) * sin( this->phase.z() + 2.0 * M_PI * z / this->waveLength.z() );
   if( XDiffOrder == 2 && YDiffOrder == 0 && ZDiffOrder == 0)
      return -4.0 * M_PI * M_PI / ( this->waveLength.x() * this->waveLength.x() ) * this->amplitude * sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) * sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) * sin( this->phase.z() + 2.0 * M_PI * z / this->waveLength.z() );
   if( XDiffOrder == 0 && YDiffOrder == 1 && ZDiffOrder == 0)
      return 2.0 * M_PI / this->waveLength.y() * this->amplitude * cos( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) * sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) * sin( this->phase.z() + 2.0 * M_PI * z / this->waveLength.z() );
   if( XDiffOrder == 0 && YDiffOrder == 2 && ZDiffOrder == 0)
      return -4.0 * M_PI * M_PI / ( this->waveLength.y() * this->waveLength.y() ) * this->amplitude * sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) * sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) * sin( this->phase.z() + 2.0 * M_PI * z / this->waveLength.z() );
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 1)
      return 2.0 * M_PI / this->waveLength.z() * this->amplitude * cos( this->phase.z() + 2.0 * M_PI * z / this->waveLength.z() ) * sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) * sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() );
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 2)
      return -4.0 * M_PI * M_PI / ( this->waveLength.z() * this->waveLength.z() ) * this->amplitude * sin( this->phase.z() + 2.0 * M_PI * z / this->waveLength.z() ) * sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) * sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() );
   return 0.0;
}

#endif /* TNLSINBUMPSFUNCTION_IMPL_H_ */
