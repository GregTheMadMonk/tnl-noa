/***************************************************************************
                          tnlSDFParaboloid.h  -  description
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

#ifndef TNLSDFPARABOLOID_H_
#define TNLSDFPARABOLOID_H_

#include <config/tnlParameterContainer.h>
#include <core/vectors/tnlStaticVector.h>

template< typename Real = double >
class tnlSDFParaboloidBase
{
   public:

   tnlSDFParaboloidBase();

   bool setup( const tnlParameterContainer& parameters,
           const tnlString& prefix = ""  );

   void setXCentre( const Real& waveLength );

   Real getXCentre() const;

   void setYCentre( const Real& waveLength );

   Real getYCentre() const;

   void setZCentre( const Real& waveLength );

   Real getZCentre() const;

   void setCoefficient( const Real& coefficient );

   Real getCoefficient() const;

   void setOffset( const Real& offset );

   Real getOffset() const;

   protected:

   Real xCentre, yCentre, zCentre, coefficient, offset;
};

template< int Dimensions, typename Real >
class tnlSDFParaboloid
{
};

template< typename Real >
class tnlSDFParaboloid< 1, Real > : public tnlSDFParaboloidBase< Real >
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
class tnlSDFParaboloid< 2, Real > : public tnlSDFParaboloidBase< Real >
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
class tnlSDFParaboloid< 3, Real > : public tnlSDFParaboloidBase< Real >
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
std::ostream& operator << ( std::ostream& str, const tnlSDFParaboloid < Dimensions, Real >& f )
{
   str << "tnlSDFParaboloid";
   return str;
};

template< int Dimensions,
          typename Real >
class tnlFunctionType< tnlSDFParaboloid< Dimensions, Real > >
{
   public:

      enum { Type = tnlAnalyticFunction };
};

#include <functions/tnlSDFParaboloid_impl.h>

#endif /* TNLSDFPARABOLOID_H_ */

