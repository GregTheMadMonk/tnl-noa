/***************************************************************************
                          tnlSDFSinWaveFunction.h  -  description
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

#ifndef TNLSDFSINWAVEFUNCTION_H_
#define TNLSDFSINWAVEFUNCTION_H_

#include <config/tnlParameterContainer.h>
#include <core/vectors/tnlStaticVector.h>

template< typename Real = double >
class tnlSDFSinWaveFunctionBase
{
   public:

   tnlSDFSinWaveFunctionBase();

   bool setup( const tnlParameterContainer& parameters,
           const tnlString& prefix = "" );

   void setWaveLength( const Real& waveLength );

   Real getWaveLength() const;

   void setAmplitude( const Real& amplitude );

   Real getAmplitude() const;

   void setWavesNumber( const Real& wavesNumber );

   Real getWavesNumber() const;

   void setPhase( const Real& phase );

   Real getPhase() const;

   protected:

   Real waveLength, amplitude, phase, wavesNumber;
};

template< int Dimensions, typename Real >
class tnlSDFSinWaveFunction
{
};

template< typename Real >
class tnlSDFSinWaveFunction< 1, Real > : public tnlSDFSinWaveFunctionBase< Real >
{
   public:

      enum { Dimensions = 1 };
      typedef tnlStaticVector< 1, Real > VertexType;
      typedef Real RealType;

#ifdef HAVE_NOT_CXX11
      template< int XDiffOrder,
                int YDiffOrder,
                int ZDiffOrder,
                typename Vertex >
#else
      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0,
                typename Vertex = VertexType >
#endif   
      RealType getValue( const Vertex& v,
                         const Real& time = 0.0 ) const;

};

template< typename Real >
class tnlSDFSinWaveFunction< 2, Real > : public tnlSDFSinWaveFunctionBase< Real >
{
   public:

         enum { Dimensions = 2 };
         typedef tnlStaticVector< 2, Real > VertexType;
         typedef Real RealType;

#ifdef HAVE_NOT_CXX11
      template< int XDiffOrder,
                int YDiffOrder,
                int ZDiffOrder,
                typename Vertex >
#else
      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0,
                typename Vertex = VertexType >
#endif   
      RealType getValue( const Vertex& v,
                         const Real& time = 0.0 ) const;
};

template< typename Real >
class tnlSDFSinWaveFunction< 3, Real > : public tnlSDFSinWaveFunctionBase< Real >
{
   public:

      enum { Dimensions = 3 };
      typedef tnlStaticVector< 3, Real > VertexType;
      typedef Real RealType;


#ifdef HAVE_NOT_CXX11
      template< int XDiffOrder,
                int YDiffOrder,
                int ZDiffOrder,
                typename Vertex >
#else
      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0,
                typename Vertex = VertexType >
#endif   
      RealType getValue( const Vertex& v,
                         const Real& time = 0.0 ) const;
};

#include <implementation/functions/tnlSDFSinWaveFunction_impl.h>

#endif /* TNLSDFSINWAVEFUNCTION_H_ */
