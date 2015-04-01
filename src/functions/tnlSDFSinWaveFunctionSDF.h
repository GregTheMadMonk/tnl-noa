/***************************************************************************
                          tnlSDFSinWaveFunctionSDF.h  -  description
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

template< typename Real = double >
class tnlSDFSinWaveFunctionSDFBase
{
   public:

   tnlSDFSinWaveFunctionSDFBase();

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

   Real waveLength, amplitude, phase, wavesNumber; // phase is currently being ignored
};

template< int Dimensions, typename Real >
class tnlSDFSinWaveFunctionSDF
{
};

template< typename Real >
class tnlSDFSinWaveFunctionSDF< 1, Real > : public tnlSDFSinWaveFunctionSDFBase< Real >
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
class tnlSDFSinWaveFunctionSDF< 2, Real > : public tnlSDFSinWaveFunctionSDFBase< Real >
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
class tnlSDFSinWaveFunctionSDF< 3, Real > : public tnlSDFSinWaveFunctionSDFBase< Real >
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

template< int Dimensions,
          typename Real >
std::ostream& operator << ( std::ostream& str, const tnlSDFSinWaveFunctionSDF< Dimensions, Real >& f )
{
   str << "tnlSDFSinWaveFunctionSDF";
   return str;
};

template< int Dimensions,
          typename Real >
class tnlFunctionType< tnlSDFSinWaveFunctionSDF< Dimensions, Real > >
{
   public:

      enum { Type = tnlAnalyticFunction };
};

#include <functions/tnlSDFSinWaveFunctionSDF_impl.h>

#endif /* TNLSDFSINWAVEFUNCTIONSDF_H_ */
