/***************************************************************************
                          SinBumpsSDF.h  -  description
                             -------------------
    begin                : Oct 13, 2014
    copyright            : (C) 2014 by Tomas Sobotik

 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Functions/Domain.h>

namespace TNL {
   namespace Functions {
      namespace Analytic {


template< typename Point >
class SinBumpsSDFBase : public Domain< Point::size, SpaceDomain >
{
   public:

      typedef Point PointType;
      typedef typename Point::RealType RealType;
      enum { Dimensions = PointType::size };

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

      PointType waveLength, phase, wavesNumber;
};

template< int Dimensions, typename Real >
class SinBumpsSDF
{
};

template< typename Real >
class SinBumpsSDF< 1, Real  > : public SinBumpsSDFBase< Containers::StaticVector< 1, Real > >
{
   public:

      typedef Real RealType;
      typedef Containers::StaticVector< 1, RealType > PointType;


      SinBumpsSDF();

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
class SinBumpsSDF< 2, Real > : public SinBumpsSDFBase< Containers::StaticVector< 2, Real > >
{
   public:

      typedef Real RealType;
      typedef Containers::StaticVector< 2, RealType > PointType;


      SinBumpsSDF();

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
class SinBumpsSDF< 3, Real > : public SinBumpsSDFBase< Containers::StaticVector< 3, Real > >
{
   public:

      typedef Real RealType;
      typedef Containers::StaticVector< 3, RealType > PointType;

      SinBumpsSDF();

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

template< int Dimensions,
          typename Real >
std::ostream& operator << ( std::ostream& str, const SinBumpsSDF< Dimensions, Real >& f )
{
   str << "SDF Sin Bumps SDF. function: amplitude = " << f.getAmplitude()
       << " wavelength = " << f.getWaveLength()
       << " phase = " << f.getPhase();
   return str;
}


      } // namespace Analytic
   } // namespace Functions
} // namespace TNL

#include <TNL/Functions/Analytic/SinBumpsSDF_impl.h>
