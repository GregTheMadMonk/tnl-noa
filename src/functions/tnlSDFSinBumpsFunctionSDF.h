/***************************************************************************
                          tnlSDFSinBumpsFunctionSDFSDF.h  -  description
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

#ifndef TNLSDFSINBUMPSFUNCTIONSDF_H_
#define TNLSDFSINBUMPSFUNCTIONSDF_H_

#include <config/tnlParameterContainer.h>
#include <core/vectors/tnlStaticVector.h>
#include <functions/tnlDomain.h>

template< typename Vertex >
class tnlSDFSinBumpsFunctionSDFBase : public tnlDomain< Vertex::size, SpaceDomain >
{
   public:

      typedef Vertex VertexType;
      typedef typename Vertex::RealType RealType;
      enum { Dimensions = VertexType::size };

      void setWaveLength( const VertexType& waveLength );

      const VertexType& getWaveLength() const;

      void setAmplitude( const RealType& amplitude );

      const RealType& getAmplitude() const;

      void setPhase( const VertexType& phase );

      const VertexType& getPhase() const;

      void setWavesNumber( const VertexType& wavesNumber );

      const VertexType& getWavesNumber() const;

   protected:

      RealType amplitude;

      VertexType waveLength, phase, wavesNumber;
};

template< int Dimensions, typename Real >
class tnlSDFSinBumpsFunctionSDF
{
};

template< typename Real >
class tnlSDFSinBumpsFunctionSDF< 1, Real  > : public tnlSDFSinBumpsFunctionSDFBase< tnlStaticVector< 1, Real > >
{
   public:

      typedef Real RealType;
      typedef tnlStaticVector< 1, RealType > VertexType;


      tnlSDFSinBumpsFunctionSDF();

      bool setup( const tnlParameterContainer& parameters,
                  const tnlString& prefix = "" );

#ifdef HAVE_NOT_CXX11
      template< int XDiffOrder,
                int YDiffOrder,
                int ZDiffOrder >
#else
      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
#endif
      __cuda_callable__
      RealType getPartialDerivative( const VertexType& v,
                                     const Real& time = 0.0 ) const;

   __cuda_callable__
   RealType operator()( const VertexType& v,
                        const Real& time = 0.0 ) const;

};

template< typename Real >
class tnlSDFSinBumpsFunctionSDF< 2, Real > : public tnlSDFSinBumpsFunctionSDFBase< tnlStaticVector< 2, Real > >
{
   public:

      typedef Real RealType;
      typedef tnlStaticVector< 2, RealType > VertexType;


      tnlSDFSinBumpsFunctionSDF();

      bool setup( const tnlParameterContainer& parameters,
                 const tnlString& prefix = "" );

#ifdef HAVE_NOT_CXX11
      template< int XDiffOrder,
                int YDiffOrder,
                int ZDiffOrder >
#else
      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
#endif
      __cuda_callable__
      RealType getPartialDerivative( const VertexType& v,
                                     const Real& time = 0.0 ) const;

   __cuda_callable__
   RealType operator()( const VertexType& v,
                        const Real& time = 0.0 ) const;

};

template< typename Real >
class tnlSDFSinBumpsFunctionSDF< 3, Real > : public tnlSDFSinBumpsFunctionSDFBase< tnlStaticVector< 3, Real > >
{
   public:

      typedef Real RealType;
      typedef tnlStaticVector< 3, RealType > VertexType;

      tnlSDFSinBumpsFunctionSDF();

      bool setup( const tnlParameterContainer& parameters,
                  const tnlString& prefix = "" );

#ifdef HAVE_NOT_CXX11
      template< int XDiffOrder,
                int YDiffOrder,
                int ZDiffOrder >
#else
      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
#endif
      __cuda_callable__
      RealType getPartialDerivative( const VertexType& v,
                         const Real& time = 0.0 ) const;

   __cuda_callable__
   RealType operator()( const VertexType& v,
                        const Real& time = 0.0 ) const;

};

template< int Dimensions,
          typename Real >
ostream& operator << ( ostream& str, const tnlSDFSinBumpsFunctionSDF< Dimensions, Real >& f )
{
   str << "SDF Sin Bumps SDF. function: amplitude = " << f.getAmplitude()
       << " wavelength = " << f.getWaveLength()
       << " phase = " << f.getPhase();
   return str;
}

#include <functions/tnlSDFSinBumpsFunctionSDF_impl.h>

#endif /* TNLSDFSINBUMPSFUNCTIONSDF_H_ */
