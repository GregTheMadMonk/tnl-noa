/***************************************************************************
                          SinBumps.h  -  description
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

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Functions/Domain.h>

namespace TNL {
namespace Functions {
namespace Analytic {   

template< typename Point >
class SinBumpsBase : public Domain< Point::getSize(), SpaceDomain >
{
   public:
 
      typedef Point PointType;
      typedef typename Point::RealType RealType;
      enum { Dimension = PointType::getSize() };

      void setWaveLength( const PointType& waveLength );

      const PointType& getWaveLength() const;

      void setAmplitude( const RealType& amplitude );

      const RealType& getAmplitude() const;

      void setPhase( const PointType& phase );

      const PointType& getPhase() const;

      void setWavesNumber( const PointType& wavesNumber );

      const PointType& getWavesNumber() const;

   protected:

      RealType amplitude;

      PointType waveLength, wavesNumber, phase;
};

template< int Dimension, typename Real >
class SinBumps
{
};

template< typename Real >
class SinBumps< 1, Real  > : public SinBumpsBase< Containers::StaticVector< 1, Real > >
{
   public:
 
      typedef Real RealType;
      typedef Containers::StaticVector< 1, RealType > PointType;

      SinBumps();

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" );

      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
      __cuda_callable__
      RealType getPartialDerivative( const PointType& v,
                                     const Real& time = 0.0 ) const;
 
   __cuda_callable__
   RealType operator()( const PointType& v,
                        const Real& time = 0.0 ) const;
 
};

template< typename Real >
class SinBumps< 2, Real > : public SinBumpsBase< Containers::StaticVector< 2, Real > >
{
   public:

      typedef Real RealType;
      typedef Containers::StaticVector< 2, RealType > PointType;
 

      SinBumps();

      bool setup( const Config::ParameterContainer& parameters,
                 const String& prefix = "" );

      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
      __cuda_callable__
      RealType getPartialDerivative( const PointType& v,
                                     const Real& time = 0.0 ) const;
 
   __cuda_callable__
   RealType operator()( const PointType& v,
                        const Real& time = 0.0 ) const;
 
};

template< typename Real >
class SinBumps< 3, Real > : public SinBumpsBase< Containers::StaticVector< 3, Real > >
{
   public:

      typedef Real RealType;
      typedef Containers::StaticVector< 3, RealType > PointType;

      SinBumps();

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" );

      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
      __cuda_callable__
      RealType getPartialDerivative( const PointType& v,
                         const Real& time = 0.0 ) const;
 
   __cuda_callable__
   RealType operator()( const PointType& v,
                        const Real& time = 0.0 ) const;
 
};

template< int Dimension,
          typename Real >
std::ostream& operator << ( std::ostream& str, const SinBumps< Dimension, Real >& f )
{
   str << "Sin Bumps. function: amplitude = " << f.getAmplitude()
       << " wavelength = " << f.getWaveLength()
       << " phase = " << f.getPhase();
   return str;
}

} // namespace Analytic
} // namespace Functions
} // namespace TNL

#include <TNL/Functions/Analytic/SinBumps_impl.h>
