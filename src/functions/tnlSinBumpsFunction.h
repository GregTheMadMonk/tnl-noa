/***************************************************************************
                          tnlSinBumpsFunction.h  -  description
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

#ifndef TNLSINBUMPSFUNCTION_H_
#define TNLSINBUMPSFUNCTION_H_

#include <config/tnlParameterContainer.h>
#include <core/vectors/tnlStaticVector.h>

template< typename Vertex >
class tnlSinBumpsFunctionBase
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

   protected:

      RealType amplitude;

      VertexType waveLength, phase;
};

template< int Dimensions, typename Real >
class tnlSinBumpsFunction
{
};

template< typename Real >
class tnlSinBumpsFunction< 1, Real  > : public tnlSinBumpsFunctionBase< tnlStaticVector< 1, Real > >
{
   public:

      enum { Dimensions = 1 };
      typedef tnlSinBumpsFunctionBase< tnlStaticVector< 1, Real > > BaseType;
      typedef typename BaseType::VertexType VertexType;
      typedef Real RealType;

      tnlSinBumpsFunction();

      bool setup( const tnlParameterContainer& parameters,
                const tnlString& prefix = "" );

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
#ifdef HAVE_CUDA
      __device__ __host__
#endif
      RealType getValue( const Vertex& v,
                         const Real& time = 0.0 ) const;
};

template< typename Real >
class tnlSinBumpsFunction< 2, Real > : public tnlSinBumpsFunctionBase< tnlStaticVector< 2, Real > >
{
   public:

      enum { Dimensions = 2 };
      typedef tnlSinBumpsFunctionBase< tnlStaticVector< 2, Real > > BaseType;
      typedef typename BaseType::VertexType VertexType;
      typedef Real RealType;

      tnlSinBumpsFunction();

      bool setup( const tnlParameterContainer& parameters,
                 const tnlString& prefix = "" );

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
#ifdef HAVE_CUDA
      __device__ __host__
#endif
      RealType getValue( const Vertex& v,
                         const Real& time = 0.0 ) const;
};

template< typename Real >
class tnlSinBumpsFunction< 3, Real > : public tnlSinBumpsFunctionBase< tnlStaticVector< 3, Real > >
{
   public:

      enum { Dimensions = 3 };
      typedef tnlSinBumpsFunctionBase< tnlStaticVector< 3, Real > > BaseType;
      typedef typename BaseType::VertexType VertexType;
      typedef Real RealType;

      tnlSinBumpsFunction();

      bool setup( const tnlParameterContainer& parameters,
                 const tnlString& prefix = "" );

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
#ifdef HAVE_CUDA
      __device__ __host__
#endif
      RealType getValue( const Vertex& v,
                         const Real& time = 0.0 ) const;
};

template< int Dimensions,
          typename Real >
ostream& operator << ( ostream& str, const tnlSinBumpsFunction< Dimensions, Real >& f )
{
   str << "Sin Bumps. function: amplitude = " << f.getAmplitude()
       << " wavelength = " << f.getWaveLength()
       << " phase = " << f.getPhase();
   return str;
}


#include <implementation/functions/tnlSinBumpsFunction_impl.h>


#endif /* TNLSINBUMPSFUNCTION_H_ */
