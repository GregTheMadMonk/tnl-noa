/***************************************************************************
                          tnlSinWaveFunction.h  -  description
                             -------------------
    begin                : Nov 19, 2013
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

#ifndef TNLSINWAVEFUNCTION_H_
#define TNLSINWAVEFUNCTION_H_

#include <config/tnlParameterContainer.h>
#include <core/vectors/tnlStaticVector.h>
#include <functions/tnlFunctionType.h>

template< typename Real = double >
class tnlSinWaveFunctionBase
{
   public:

   tnlSinWaveFunctionBase();

   bool setup( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" );

   void setWaveLength( const Real& waveLength );

   Real getWaveLength() const;

   void setAmplitude( const Real& amplitude );

   Real getAmplitude() const;

   void setPhase( const Real& phase );

   Real getPhase() const;

   protected:

   Real waveLength, amplitude, phase, wavesNumber;
};

template< int Dimensions, typename Real >
class tnlSinWaveFunction
{
};

template< typename Real >
class tnlSinWaveFunction< 1, Real > : public tnlSinWaveFunctionBase< Real >
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
#ifdef HAVE_CUDA
      __device__ __host__
#endif
      RealType getValue( const Vertex& v,
                         const Real& time = 0.0 ) const;

};

template< typename Real >
class tnlSinWaveFunction< 2, Real > : public tnlSinWaveFunctionBase< Real >
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
#ifdef HAVE_CUDA
      __device__ __host__
#endif
      RealType getValue( const Vertex& v,
                         const Real& time = 0.0 ) const;
};

template< typename Real >
class tnlSinWaveFunction< 3, Real > : public tnlSinWaveFunctionBase< Real >
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
#ifdef HAVE_CUDA
      __device__ __host__
#endif
      RealType getValue( const Vertex& v,
                         const Real& time = 0.0 ) const;
};

template< int Dimensions,
          typename Real >
ostream& operator << ( ostream& str, const tnlSinWaveFunction< Dimensions, Real >& f )
{
   str << "Sin Wave. function: amplitude = " << f.getAmplitude()
       << " wavelength = " << f.getWaveLength()
       << " phase = " << f.getPhase();
   return str;
}

template< int FunctionDimensions,
          typename Real >
class tnlFunctionType< tnlSinWaveFunction< FunctionDimensions, Real > >
{
   public:

      enum { Type = tnlAnalyticFunction };
};

#include <functions/tnlSinWaveFunction_impl.h>

#endif /* TNLSINWAVEFUNCTION_H_ */
