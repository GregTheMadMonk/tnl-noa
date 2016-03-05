/***************************************************************************
                          tnlSDFSinWaveFunctionSDFSDF.h  -  description
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

#ifndef TNLSDFSINWAVEFUNCTIONSDF_H_
#define TNLSDFSINWAVEFUNCTIONSDF_H_

#include <config/tnlParameterContainer.h>
#include <core/vectors/tnlStaticVector.h>
#include <functions/tnlDomain.h>

template< int dimensions,
          typename Real = double >
class tnlSDFSinWaveFunctionSDFBase : public tnlDomain< dimensions, SpaceDomain >
{
   public:

   tnlSDFSinWaveFunctionSDFBase();

   bool setup( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" );

   void setWaveLength( const Real& waveLength );

   Real getWaveLength() const;

   void setAmplitude( const Real& amplitude );

   Real getAmplitude() const;

   void setPhase( const Real& phase );

   Real getPhase() const;

   void setWavesNumber( const Real& wavesNumber );

   Real getWavesNumber() const;

   protected:

   Real waveLength, amplitude, phase, wavesNumber;
};

template< int Dimensions, typename Real >
class tnlSDFSinWaveFunctionSDF
{
};

template< typename Real >
class tnlSDFSinWaveFunctionSDF< 1, Real > : public tnlSDFSinWaveFunctionSDFBase< 1, Real >
{
   public:

      typedef Real RealType;
      typedef tnlStaticVector< 1, RealType > VertexType;

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
class tnlSDFSinWaveFunctionSDF< 2, Real > : public tnlSDFSinWaveFunctionSDFBase< 2, Real >
{
   public:

      typedef Real RealType;
      typedef tnlStaticVector< 2, RealType > VertexType;

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
class tnlSDFSinWaveFunctionSDF< 3, Real > : public tnlSDFSinWaveFunctionSDFBase< 3, Real >
{
   public:

      typedef Real RealType;
      typedef tnlStaticVector< 3, RealType > VertexType;



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
ostream& operator << ( ostream& str, const tnlSDFSinWaveFunctionSDF< Dimensions, Real >& f )
{
   str << "SDF Sin Wave SDF. function: amplitude = " << f.getAmplitude()
       << " wavelength = " << f.getWaveLength()
       << " phase = " << f.getPhase()
       << " # of waves = " << f.getWavesNumber();
   return str;
}

#include <functions/tnlSDFSinWaveFunctionSDF_impl.h>

#endif /* TNLSDFSINWAVEFUNCTIONSDF_H_ */
