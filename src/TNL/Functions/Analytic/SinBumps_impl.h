/***************************************************************************
                          SinBumps_impl.h  -  description
                             -------------------
    begin                : Dec 5, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/****
 * Tomas Sobotik
 */
#pragma once

#include <TNL/Functions/Analytic/SinBumps.h>

namespace TNL {
namespace Functions {
namespace Analytic {   

template< typename Vertex >
void SinBumpsBase< Vertex >::setWaveLength( const Vertex& waveLength )
{
   this->waveLength = waveLength;
}

template< typename Vertex >
const Vertex& SinBumpsBase< Vertex >::getWaveLength() const
{
   return this->waveLength;
}

template< typename Vertex >
void SinBumpsBase< Vertex >::setAmplitude( const typename Vertex::RealType& amplitude )
{
   this->amplitude = amplitude;
}

template< typename Vertex >
const typename Vertex::RealType& SinBumpsBase< Vertex >::getAmplitude() const
{
   return this->amplitude;
}

template< typename Vertex >
void SinBumpsBase< Vertex >::setPhase( const Vertex& phase )
{
   this->phase = phase;
}

template< typename Vertex >
const Vertex& SinBumpsBase< Vertex >::getPhase() const
{
   return this->phase;
}

template< typename Vertex >
void SinBumpsBase< Vertex >::setWavesNumber( const Vertex& wavesNumber )
{
   this->wavesNumber = wavesNumber;
}

template< typename Vertex >
const Vertex& SinBumpsBase< Vertex >::getWavesNumber() const
{
   return this->wavesNumber;
}

/***
 * 1D
 */
template< typename Real >
SinBumps< 1, Real >::SinBumps()
{
}

template< typename Real >
bool SinBumps< 1, Real >::setup( const Config::ParameterContainer& parameters,
                                           const String& prefix )
{
   this->amplitude = parameters.getParameter< double >( prefix + "amplitude" );
   this->waveLength.x() = parameters.getParameter< double >( prefix + "wave-length-x" );
   this->phase.x() = parameters.getParameter< double >( prefix + "phase-x" );
   this->wavesNumber.x() = ceil( parameters.getParameter< double >( prefix+"waves-number-x" ) );
   return true;
}


template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__
Real
SinBumps< 1, Real >::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   
   const RealType& x = v.x();
   const RealType xp = ::fabs( x ) + sign( x ) * this->phase.x() * this->waveLength.x() / (2.0*M_PI);
   if( this->wavesNumber.x() != 0.0 && xp > this->waveLength.x() * this->wavesNumber.x() )
      return 0.0;
  
   if( XDiffOrder == 0 )
      return this->amplitude * ::sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() );
   if( XDiffOrder == 1 )
      return 2.0 * M_PI / this->waveLength.x() * this->amplitude * ::cos( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() );
   if( XDiffOrder == 2 )
      return -4.0 * M_PI * M_PI / ( this->waveLength.x() * this->waveLength.x() ) * this->amplitude * ::sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() );
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
SinBumps< 1, Real >::
operator()( const VertexType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}


/****
 * 2D
 */
template< typename Real >
SinBumps< 2, Real >::SinBumps()
{
}

template< typename Real >
bool SinBumps< 2, Real >::setup( const Config::ParameterContainer& parameters,
                                            const String& prefix )
{
   this->amplitude = parameters.getParameter< double >( prefix + "amplitude" );
   this->waveLength.x() = parameters.getParameter< double >( prefix + "wave-length-x" );
   this->waveLength.y() = parameters.getParameter< double >( prefix + "wave-length-y" );
   this->phase.x() = parameters.getParameter< double >( prefix + "phase-x" );
   this->phase.y() = parameters.getParameter< double >( prefix + "phase-y" );
   this->wavesNumber.x() = ceil( parameters.getParameter< double >( prefix+"waves-number-x" ) );
   this->wavesNumber.y() = ceil( parameters.getParameter< double >( prefix+"waves-number-y" ) );
   return true;
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__
Real
SinBumps< 2, Real>::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   if( ZDiffOrder != 0 )
      return 0.0;

   const RealType& x = v.x();
   const RealType& y = v.y();
   const RealType xp = ::fabs( x ) + sign( x ) * this->phase.x() * this->waveLength.x() / (2.0*M_PI);
   const RealType yp = ::fabs( y ) + sign( y ) * this->phase.y() * this->waveLength.y() / (2.0*M_PI);
   //std::cerr << "this->wavesNumber.x() = " << this->wavesNumber.x() << "fabs( x ) = " << fabs( x ) << " 2.0*M_PI * this->waveLength.x() * this->wavesNumber.x() = " << 2.0*M_PI * this->waveLength.x() * this->wavesNumber.x() << std::endl;
   if( ( this->wavesNumber.x() != 0.0 && xp > this->waveLength.x() * this->wavesNumber.x() ) ||
       ( this->wavesNumber.y() != 0.0 && yp > this->waveLength.y() * this->wavesNumber.y() ) )
      return 0.0;
   
   if( XDiffOrder == 0 && YDiffOrder == 0 )
      return this->amplitude *
             ::sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) *
             ::sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() );
   if( XDiffOrder == 1 && YDiffOrder == 0 )
      return 2.0 * M_PI / this->waveLength.x() * this->amplitude * ::cos( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) * ::sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() );
   if( XDiffOrder == 2 && YDiffOrder == 0 )
      return -4.0 * M_PI * M_PI / ( this->waveLength.x() * this->waveLength.x() ) * this->amplitude * ::sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) * ::sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() );
   if( XDiffOrder == 0 && YDiffOrder == 1 )
      return 2.0 * M_PI / this->waveLength.y() * this->amplitude * ::cos( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) * ::sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() );
   if( XDiffOrder == 0 && YDiffOrder == 2 )
      return -4.0 * M_PI * M_PI / ( this->waveLength.y() * this->waveLength.y() ) * this->amplitude * ::sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) * ::sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() );
   if( XDiffOrder == 1 && YDiffOrder == 1 )
      return 4.0 * M_PI * M_PI / ( this->waveLength.x() * this->waveLength.y() ) * this->amplitude * ::cos( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) * ::cos( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() );
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
SinBumps< 2, Real >::
operator()( const VertexType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}

/****
 * 3D
 */
template< typename Real >
SinBumps< 3, Real >::SinBumps()
{
}

template< typename Real >
bool SinBumps< 3, Real >::setup( const Config::ParameterContainer& parameters,
                                           const String& prefix )
{
   this->amplitude = parameters.getParameter< double >( prefix + "amplitude" );
   this->waveLength.x() = parameters.getParameter< double >( prefix + "wave-length-x" );
   this->waveLength.y() = parameters.getParameter< double >( prefix + "wave-length-y" );
   this->waveLength.z() = parameters.getParameter< double >( prefix + "wave-length-z" );
   this->phase.x() = parameters.getParameter< double >( prefix + "phase-x" );
   this->phase.y() = parameters.getParameter< double >( prefix + "phase-y" );
   this->phase.z() = parameters.getParameter< double >( prefix + "phase-z" );
   this->wavesNumber.x() = ceil( parameters.getParameter< double >( prefix+"waves-number-x" ) );
   this->wavesNumber.y() = ceil( parameters.getParameter< double >( prefix+"waves-number-y" ) );
   this->wavesNumber.z() = ceil( parameters.getParameter< double >( prefix+"waves-number-z" ) );
   return true;
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__
Real
SinBumps< 3, Real >::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   const RealType& z = v.z();
   
   const RealType xp = ::fabs( x ) + sign( x ) * this->phase.x() * this->waveLength.x() / (2.0*M_PI);
   const RealType yp = ::fabs( y ) + sign( y ) * this->phase.y() * this->waveLength.y() / (2.0*M_PI);
   const RealType zp = ::fabs( z ) + sign( z ) * this->phase.z() * this->waveLength.z() / (2.0*M_PI);

   if( ( this->wavesNumber.x() != 0.0 && xp > this->waveLength.x() * this->wavesNumber.x() ) ||
       ( this->wavesNumber.y() != 0.0 && yp > this->waveLength.y() * this->wavesNumber.y() ) ||
       ( this->wavesNumber.z() != 0.0 && zp > this->waveLength.z() * this->wavesNumber.z() ) )
      return 0.0;
   
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0)
      return this->amplitude *
             ::sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) *
             ::sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) *
             ::sin( this->phase.z() + 2.0 * M_PI * z / this->waveLength.z() );
   if( XDiffOrder == 1 && YDiffOrder == 0 && ZDiffOrder == 0)
      return 2.0 * M_PI / this->waveLength.x() * this->amplitude * ::cos( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) * ::sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) * ::sin( this->phase.z() + 2.0 * M_PI * z / this->waveLength.z() );
   if( XDiffOrder == 2 && YDiffOrder == 0 && ZDiffOrder == 0)
      return -4.0 * M_PI * M_PI / ( this->waveLength.x() * this->waveLength.x() ) * this->amplitude * ::sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) * ::sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) * ::sin( this->phase.z() + 2.0 * M_PI * z / this->waveLength.z() );
   if( XDiffOrder == 0 && YDiffOrder == 1 && ZDiffOrder == 0)
      return 2.0 * M_PI / this->waveLength.y() * this->amplitude * ::cos( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) * ::sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) * ::sin( this->phase.z() + 2.0 * M_PI * z / this->waveLength.z() );
   if( XDiffOrder == 0 && YDiffOrder == 2 && ZDiffOrder == 0)
      return -4.0 * M_PI * M_PI / ( this->waveLength.y() * this->waveLength.y() ) * this->amplitude * ::sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) * ::sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) * ::sin( this->phase.z() + 2.0 * M_PI * z / this->waveLength.z() );
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 1)
      return 2.0 * M_PI / this->waveLength.z() * this->amplitude * ::cos( this->phase.z() + 2.0 * M_PI * z / this->waveLength.z() ) * ::sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) * ::sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() );
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 2)
      return -4.0 * M_PI * M_PI / ( this->waveLength.z() * this->waveLength.z() ) * this->amplitude * ::sin( this->phase.z() + 2.0 * M_PI * z / this->waveLength.z() ) * ::sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) * ::sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() );
   if( XDiffOrder == 1 && YDiffOrder == 1 && ZDiffOrder == 0)
      return 4.0 * M_PI * M_PI / ( this->waveLength.x() * this->waveLength.y() ) * this->amplitude * ::cos( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) * ::cos( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) * ::sin( this->phase.z() + 2.0 * M_PI * z / this->waveLength.z() );
   if( XDiffOrder == 1 && YDiffOrder == 0 && ZDiffOrder == 1)
      return 4.0 * M_PI * M_PI / ( this->waveLength.x() * this->waveLength.z() ) * this->amplitude * ::cos( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) * ::sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) * ::cos( this->phase.z() + 2.0 * M_PI * z / this->waveLength.z() );
   if( XDiffOrder == 0 && YDiffOrder == 1 && ZDiffOrder == 1)
      return 4.0 * M_PI * M_PI / ( this->waveLength.y() * this->waveLength.z() ) * this->amplitude * ::sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() ) * ::cos( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) * ::cos( this->phase.z() + 2.0 * M_PI * z / this->waveLength.z() );
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
SinBumps< 3, Real >::
operator()( const VertexType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}

} // namespace Analytic
} // namespace Functions
} // namespace TNL

