/***************************************************************************
                          tnlExplicitTimeStepper.h  -  description
                             -------------------
    begin                : Jan 16, 2013
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

#ifndef TNLGRID_H_
#define TNLGRID_H_

#include <core/tnlObject.h>
#include <core/tnlHost.h>
#include <core/tnlTuple.h>
#include <core/vectors/tnlVector.h>
#include <mesh/tnlIdenticalGridGeometry.h>

template< int Dimensions,
          typename Real = double,
          typename Device = tnlHost,
          typename Index = int,
          template< int, typename, typename, typename > class Geometry = tnlIdenticalGridGeometry >
class tnlGrid : public tnlObject
{
};

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
class tnlGrid< 1, Real, Device, Index, Geometry > : public tnlObject
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef Geometry< 1, Real, Device, Index > GeometryType;
   typedef tnlTuple< 1, Real > VertexType;
   typedef tnlTuple< 1, Index > CoordinatesType;

   enum { Dimensions = 1};

   tnlGrid();

   static tnlString getTypeStatic();

   tnlString getTypeVirtual() const;

   tnlString getType() const;

   void setDimensions( const Index xSize );

   void setDimensions( const CoordinatesType& );

   const CoordinatesType& getDimensions() const;

   void setOrigin( const VertexType& origin );

   const VertexType& getOrigin() const;

   void setProportions( const VertexType& proportions );

   const VertexType& getProportions() const;

   void setParametricStep( const VertexType& spaceStep );

   const VertexType& getParametricStep() const;

   Index getElementIndex( const Index i ) const;

   Index getDofs() const;

   //! Method for saving the object to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   template< typename MeshFunction >
   bool write( const MeshFunction& function,
               const tnlString& fileName,
               const tnlString& format ) const;

   protected:

   tnlTuple< 1, IndexType > dimensions;

   tnlTuple< 1, RealType > origin, proportions;

   IndexType dofs;

   GeometryType geometry;

};

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
class tnlGrid< 2, Real, Device, Index, Geometry > : public tnlObject
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef Geometry< 2, Real, Device, Index > GeometryType;
   typedef tnlTuple< 2, Real > VertexType;
   typedef tnlTuple< 2, Index > CoordinatesType;
   enum { Dimensions = 2};

   tnlGrid();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   bool setDimensions( const Index xSize, const Index ySize );

   bool setDimensions( const CoordinatesType& );

   const CoordinatesType& getDimensions() const;

   void setOrigin( const VertexType& origin );

   const VertexType& getOrigin() const;

   void setProportions( const VertexType& proportions );

   const VertexType& getProportions() const;

   void setParametricStep( const VertexType& spaceStep );

   const VertexType& getParametricStep() const;

   Index getElementIndex( const Index i,
                          const Index j ) const;

   Index getEdgeIndex( const Index i,
                       const Index j,
                       const Index dx,
                       const Index dy ) const;

   template< int dx, int dy >
   Index getVertexIndex( const Index i,
                         const Index j ) const;

   void refresh();

   void getElementCoordinates( const Index i,
                               CoordinatesType& coordinates ) const;

   Index getElementNeighbour( const Index Element,
                              const Index dx,
                              const Index dy ) const;

   Index getDofs() const;

   Index getNumberOfEdges() const;

   Index getNumberOfVertices() const;

   GeometryType& getGeometry();

   const GeometryType& getGeometry() const;

   void getElementCenter( const CoordinatesType& coordinates,
                          VertexType& center ) const;

   Real getElementMeasure( const CoordinatesType& coordinates ) const;

   template< int dx, int dy >
   Real getDualElementMeasure( const CoordinatesType& coordinates ) const;

   template< int dx, int dy >
   void getEdgeNormal( const CoordinatesType& elementCoordinates,
                       VertexType& normal ) const;

   template< int dx, int dy >
   void getVertex( const CoordinatesType& elementCoordinates,
                   VertexType& vertex ) const;

   //! Method for saving the object to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   bool writeMesh( const tnlString& fileName,
                   const tnlString& format ) const;

   template< typename MeshFunction >
   bool write( const MeshFunction& function,
               const tnlString& fileName,
               const tnlString& format ) const;

   protected:

   CoordinatesType dimensions;

   VertexType origin;

   GeometryType geometry;

   IndexType dofs;

   tnlVector< Real, Device, Index > elementsMeasure, dualElementsMeasure;
   tnlVector< VertexType, Device, Index > edgeNormals, vertices, elementCenters;

};

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
class tnlGrid< 3, Real, Device, Index, Geometry > : public tnlObject
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef Geometry< 2, Real, Device, Index > GeometryType;
   typedef tnlTuple< 3, Real > VertexType;
   typedef tnlTuple< 3, Index > CoordinatesType;
   enum { Dimensions = 3};

   tnlGrid();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   void setDimensions( const Index xSize, const Index ySize, const Index zSize );

   void setDimensions( const CoordinatesType& );

   const CoordinatesType& getDimensions() const;

   void setOrigin( const VertexType& origin );

   const VertexType& getOrigin() const;

   void setProportions( const VertexType& proportions );

   const VertexType& getProportions() const;

   void setParametricStep( const VertexType& spaceStep );

   const VertexType& getParametricStep() const;

   Index getElementIndex( const Index i, const Index j, const Index k ) const;

   Index getDofs() const;

   //! Method for saving the object to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   template< typename MeshFunction >
   bool write( const MeshFunction& function,
               const tnlString& fileName,
               const tnlString& format ) const;

   protected:

   tnlTuple< 3, IndexType > dimensions;

   tnlTuple< 3, RealType > origin, proportions;

   IndexType dofs;

};

#include <implementation/mesh/tnlGrid1D_impl.h>
#include <implementation/mesh/tnlGrid2D_impl.h>
#include <implementation/mesh/tnlGrid3D_impl.h>


#endif /* TNLGRID_H_ */
