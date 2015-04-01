/***************************************************************************
                          tnlSDFSinBumpsFunctionSDF.h  -  description
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

template< typename Vertex >
class tnlSDFSinBumpsFunctionSDFBase
{
   public:

    typedef Vertex VertexType;
    typedef typename Vertex::RealType RealType;
    enum { Dimensions = VertexType::size };

   void setWaveLength( const VertexType& waveLength );

   const VertexType& getWaveLength() const;

   void setWavesNumber( const VertexType& wavesNumber );

   const VertexType& getWavesNumber() const;

   void setAmplitude( const RealType& amplitude );

   const RealType& getAmplitude() const;

   void setPhase( const VertexType& phase );

   const VertexType& getPhase() const;

   protected:

   RealType amplitude;

   VertexType waveLength, phase, wavesNumber;
};

template< int Dimensions, typename Real >
class tnlSDFSinBumpsFunctionSDF
{
};

template< typename Real >
class tnlSDFSinBumpsFunctionSDF< 1, Real > : public tnlSDFSinBumpsFunctionSDFBase< tnlStaticVector< 1, Real> >
{
   public:

   enum { Dimensions = 1 };
   typedef tnlSinBumpsFunctionBase< tnlStaticVector< 1, Real > > BaseType;
   typedef typename BaseType::VertexType VertexType;
   typedef Real RealType;

   tnlSDFSinBumpsFunctionSDF();

   bool setup( const tnlParameterContainer& parameters,
           const tnlString& prefix = "" );

#ifdef HAVE_NOT_CXX11
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex = VertexType >
   RealType getValue( const Vertex& v,
           const Real& time = 0.0  ) const;
#else
   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0,
             typename Vertex = VertexType >
   RealType getValue( const Vertex& v,
           const Real& time = 0.0  ) const;
#endif
};

template< typename Real >
class tnlSDFSinBumpsFunctionSDF< 2, Real > : public tnlSDFSinBumpsFunctionSDFBase< tnlStaticVector< 2, Real> >
{
   public:

   enum { Dimensions = 2 };
   typedef tnlSinBumpsFunctionBase< tnlStaticVector< 2, Real > > BaseType;
   typedef typename BaseType::VertexType VertexType;
   typedef Real RealType;

   tnlSDFSinBumpsFunctionSDF();

   bool setup( const tnlParameterContainer& parameters,
           const tnlString& prefix = "" );

#ifdef HAVE_NOT_CXX11
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex = VertexType >
   RealType getValue( const Vertex& v,
           const Real& time = 0.0  ) const;
#else
   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0,
             typename Vertex = VertexType >
   RealType getValue( const Vertex& v,
           const Real& time = 0.0  ) const;
#endif
};

template< typename Real >
class tnlSDFSinBumpsFunctionSDF< 3, Real > : public tnlSDFSinBumpsFunctionSDFBase< tnlStaticVector< 3, Real> >
{
   public:

   enum { Dimensions = 3 };
   typedef tnlSinBumpsFunctionBase< tnlStaticVector< 3, Real > > BaseType;
   typedef typename BaseType::VertexType VertexType;
   typedef Real RealType;

   tnlSDFSinBumpsFunctionSDF();

   bool setup( const tnlParameterContainer& parameters,
           const tnlString& prefix = "" );

#ifdef HAVE_NOT_CXX11
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex = VertexType >
   RealType getValue( const Vertex& v,
           const Real& time = 0.0  ) const;
#else
   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0,
             typename Vertex = VertexType >
   RealType getValue( const Vertex& v,
           const Real& time = 0.0  ) const;
#endif
};

#include <implementation/functions/tnlSDFSinBumpsFunctionSDF_impl.h>

#endif /* TNLSDFSINBUMPSFUNCTIONSDF_H_ */
