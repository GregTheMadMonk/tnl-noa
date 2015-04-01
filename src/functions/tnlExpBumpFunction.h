/***************************************************************************
                          tnlExpBumpFunction.h  -  description
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

#ifndef TNLEXPBUMPFUNCTION_H_
#define TNLEXPBUMPFUNCTION_H_

#include <config/tnlParameterContainer.h>
#include <core/vectors/tnlStaticVector.h>
#include <functions/tnlFunctionType.h>

template< typename Real >
class tnlExpBumpFunctionBase
{
   public:

      typedef Real RealType;

      bool setup( const tnlParameterContainer& parameters,
                 const tnlString& prefix = "" );

      void setAmplitude( const RealType& amplitude );

      const RealType& getAmplitude() const;

      void setSigma( const RealType& sigma );

      const RealType& getSigma() const;

   protected:

      RealType amplitude, sigma;
};

template< int Dimensions,
          typename Real >
class tnlExpBumpFunction
{
};

template< typename Real >
class tnlExpBumpFunction< 1, Real > : public tnlExpBumpFunctionBase< Real >
{
   public:

      enum { Dimensions = 1 };
      typedef Real RealType;
      typedef tnlStaticVector< Dimensions, Real > VertexType;

      static tnlString getType();

      tnlExpBumpFunction();

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
class tnlExpBumpFunction< 2, Real > : public tnlExpBumpFunctionBase< Real >
{
   public:

      enum { Dimensions = 2 };
      typedef Real RealType;
      typedef tnlStaticVector< Dimensions, Real > VertexType;

      static tnlString getType();

      tnlExpBumpFunction();

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
class tnlExpBumpFunction< 3, Real > : public tnlExpBumpFunctionBase< Real >
{
   public:

      enum { Dimensions = 3 };
      typedef Real RealType;
      typedef tnlStaticVector< Dimensions, Real > VertexType;

      static tnlString getType();

      tnlExpBumpFunction();

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
ostream& operator << ( ostream& str, const tnlExpBumpFunction< Dimensions, Real >& f )
{
   str << "ExpBump. function: amplitude = " << f.getAmplitude() << " sigma = " << f.getSigma();
   return str;
}

template< int FunctionDimensions,
          typename Real >
class tnlFunctionType< tnlExpBumpFunction< FunctionDimensions, Real > >
{
   public:

      enum { Type = tnlAnalyticFunction };
};


#include <functions/tnlExpBumpFunction_impl.h>


#endif /* TNLEXPBUMPFUNCTION_H_ */
