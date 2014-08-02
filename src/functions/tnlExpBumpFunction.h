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

template< typename Real >
class tnlExpBumpFunctionBase
{
   public:

   typedef Real RealType;

   bool init( const tnlParameterContainer& parameters );

   void setAmplitude( const RealType& amplitude );

   const RealType& getAmplitude() const;

   void setSigma( const RealType& sigma );

   const RealType& getSigma() const;


   protected:

   RealType amplitude, sigma;
};

template< int Dimensions, typename Vertex = tnlStaticVector< Dimensions, double >, typename Device = tnlHost >
class tnlExpBumpFunction
{
};

template< typename Vertex, typename Device >
class tnlExpBumpFunction< 1, Vertex, Device > : public tnlExpBumpFunctionBase< typename Vertex::RealType >
{
   public:

   enum { Dimensions = 1 };
   typedef Vertex VertexType;
   typedef typename VertexType::RealType RealType;

   tnlExpBumpFunction();

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
class tnlExpBumpFunction< 2, Vertex, Device > : public tnlExpBumpFunctionBase< typename Vertex::RealType >
{
   public:

   enum { Dimensions = 2 };
   typedef Vertex VertexType;
   typedef typename VertexType::RealType RealType;

   tnlExpBumpFunction();

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
class tnlExpBumpFunction< 3, Vertex, Device > : public tnlExpBumpFunctionBase< typename Vertex::RealType >
{
   public:

   enum { Dimensions = 3 };
   typedef Vertex VertexType;
   typedef typename VertexType::RealType RealType;

   tnlExpBumpFunction();

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

#include <implementation/functions/tnlExpBumpFunction_impl.h>


#endif /* TNLEXPBUMPFUNCTION_H_ */
