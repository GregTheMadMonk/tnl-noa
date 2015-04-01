/***************************************************************************
                          tnlSDFGridValue.h  -  description
                             -------------------
    begin                : Nov 22, 2014
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

#ifndef TNLSDFGRIDVALUE_H_
#define TNLSDFGRIDVALUE_H_

#include <config/tnlParameterContainer.h>
#include <core/vectors/tnlStaticVector.h>
#include <core/vectors/tnlVector.h>
#include <core/vectors/tnlSharedVector.h>
#include <mesh/tnlGrid.h>

template< typename Real = double >
class tnlSDFGridValueBase
{
   public:

   tnlSDFGridValueBase();


};

template< typename Mesh, int Dimensions, typename Real >
class tnlSDFGridValue
{
};

template< typename Mesh, typename Real >
class tnlSDFGridValue< Mesh, 1, Real > : public tnlSDFGridValueBase< Real >
{
   public:

   enum { Dimensions = 1 };
   typedef tnlStaticVector< 1, Real > VertexType;
   typedef Real RealType;
	typedef typename Mesh::RealType RealType2;
	typedef typename Mesh::DeviceType DeviceType;
	typedef typename Mesh::IndexType IndexType;
	typedef tnlVector< RealType2, DeviceType, IndexType> DofVectorType;
   typedef typename Mesh::CoordinatesType CoordinatesType;

   bool setup( const tnlParameterContainer& parameters,
           const tnlString& prefix = ""  );

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

   private:

   VertexType origin;
   RealType2 hx;
   Mesh mesh;
   DofVectorType u;

};

template< typename Mesh, typename Real >
class tnlSDFGridValue< Mesh, 2, Real > : public tnlSDFGridValueBase< Real >
{
   public:

   enum { Dimensions = 2 };
   typedef tnlStaticVector< 2, Real > VertexType;
   typedef Real RealType;
	typedef typename Mesh::RealType RealType2;
	typedef typename Mesh::DeviceType DeviceType;
	typedef typename Mesh::IndexType IndexType;
	typedef tnlVector< RealType2, DeviceType, IndexType> DofVectorType;
   typedef typename Mesh::CoordinatesType CoordinatesType;

   bool setup( const tnlParameterContainer& parameters,
           const tnlString& prefix = ""  );

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

   private:

   CoordinatesType dimensions;
   VertexType origin;
   RealType2 hx, hy;
   Mesh mesh;
   DofVectorType u;

};

template< typename Mesh, typename Real >
class tnlSDFGridValue< Mesh, 3, Real > : public tnlSDFGridValueBase< Real >
{
   public:

   enum { Dimensions = 3 };
   typedef tnlStaticVector< 3, Real > VertexType;
   typedef Real RealType;
	typedef typename Mesh::RealType RealType2;
	typedef typename Mesh::DeviceType DeviceType;
	typedef typename Mesh::IndexType IndexType;
	typedef tnlVector< RealType2, DeviceType, IndexType> DofVectorType;
   typedef typename Mesh::CoordinatesType CoordinatesType;

   bool setup( const tnlParameterContainer& parameters,
           const tnlString& prefix = ""  );

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

   private:

   CoordinatesType dimensions;
   VertexType origin;
   RealType2 hx, hy, hz;
   Mesh mesh;
   DofVectorType u;

};

#include <implementation/functions/tnlSDFGridValue_impl.h>

#endif /* TNLSDFGRIDVALUE_H_ */

