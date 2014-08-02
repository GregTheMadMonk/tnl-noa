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

template< int Dimensions, typename Vertex = tnlStaticVector< Dimensions, double >, typename Device = tnlHost >
class tnlSinBumpsFunctionBase
{
   public:

   typedef Vertex VertexType;
   typedef typename VertexType::RealType RealType;

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

template< int Dimensions, typename Vertex = tnlStaticVector< Dimensions, double >, typename Device = tnlHost >
class tnlSinBumpsFunction
{
};

template< typename Vertex, typename Device >
class tnlSinBumpsFunction< 1, Vertex, Device > : public tnlSinBumpsFunctionBase< 1, Vertex, Device >
{
   public:

   enum { Dimensions = 1 };
   typedef Vertex VertexType;
   typedef typename VertexType::RealType RealType;

   tnlSinBumpsFunction();

   bool init( const tnlParameterContainer& parameters );

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
class tnlSinBumpsFunction< 2, Vertex, Device > : public tnlSinBumpsFunctionBase< 2, Vertex, Device >
{
   public:

   enum { Dimensions = 2 };
   typedef Vertex VertexType;
   typedef typename VertexType::RealType RealType;

   tnlSinBumpsFunction();

   bool init( const tnlParameterContainer& parameters );

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
class tnlSinBumpsFunction< 3, Vertex, Device > : public tnlSinBumpsFunctionBase< 3, Vertex, Device >
{
   public:

   enum { Dimensions = 3 };
   typedef Vertex VertexType;
   typedef typename VertexType::RealType RealType;

   tnlSinBumpsFunction();

   bool init( const tnlParameterContainer& parameters );

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

#include <implementation/functions/tnlSinBumpsFunction_impl.h>


#endif /* TNLSINBUMPSFUNCTION_H_ */
