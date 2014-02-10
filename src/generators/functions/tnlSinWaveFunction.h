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

template< typename Real = double >
class tnlSinWaveFunctionBase
{
   public:

   tnlSinWaveFunctionBase();

   bool init( const tnlParameterContainer& parameters );

   void setWaveLength( const Real& waveLength );

   Real getWaveLength() const;

   void setAmplitude( const Real& amplitude );

   Real getAmplitude() const;

   void setPhase( const Real& phase );

   Real getPhase() const;

   protected:

   Real waveLength, amplitude, phase;
};

template< int Dimensions, typename Vertex = tnlStaticVector< Dimensions, double >, typename Device = tnlHost >
class tnlSinWaveFunction
{
};

template< typename Vertex, typename Device >
class tnlSinWaveFunction< 1, Vertex, Device > : public tnlSinWaveFunctionBase< typename Vertex::RealType >
{
   public:

   enum { Dimensions = 1 };
   typedef Vertex VertexType;
   typedef typename VertexType::RealType RealType;

#ifdef HAVE_NOT_CXX11
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
   RealType getF( const VertexType& v ) const;
#else
   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0 >
   RealType getF( const VertexType& v ) const;
#endif   

};

template< typename Vertex, typename Device >
class tnlSinWaveFunction< 2, Vertex, Device > : public tnlSinWaveFunctionBase< typename Vertex::RealType >
{
   public:

   enum { Dimensions = 2 };
   typedef Vertex VertexType;
   typedef typename VertexType::RealType RealType;

#ifdef HAVE_NOT_CXX11
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
   RealType getF( const VertexType& v ) const;
#else
   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0 >
   RealType getF( const VertexType& v ) const;
#endif   
};

template< typename Vertex, typename Device >
class tnlSinWaveFunction< 3, Vertex, Device > : public tnlSinWaveFunctionBase< typename Vertex::RealType >
{
   public:

   enum { Dimensions = 3 };
   typedef Vertex VertexType;
   typedef typename VertexType::RealType RealType;

#ifdef HAVE_NOT_CXX11
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
   RealType getF( const VertexType& v ) const;
#else
   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0 >
   RealType getF( const VertexType& v ) const;
#endif   
};

#include <implementation/generators/functions/tnlSinWaveFunction_impl.h>

#endif /* TNLSINWAVEFUNCTION_H_ */
