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
#include <core/vectors/tnlStaticVector.h>
#include <core/vectors/tnlVector.h>

template< int Dimensions,
          typename Real = double,
          typename Device = tnlHost,
          typename Index = int >
class tnlGrid : public tnlObject
{
};

template< typename Real,
          typename Device,
          typename Index >
class tnlGrid< 1, Real, Device, Index > : public tnlObject
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlStaticVector< 1, Real > VertexType;
   typedef tnlStaticVector< 1, Index > CoordinatesType;

   enum { Dimensions = 1};

   tnlGrid();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   bool setDimensions( const Index xSize );

   bool setDimensions( const CoordinatesType& );

   const CoordinatesType& getDimensions() const;

   void setOrigin( const VertexType& origin );

   const VertexType& getOrigin() const;

   void setProportions( const VertexType& proportions );

   const VertexType& getProportions() const;

   //void setParametricStep( const VertexType& spaceStep );

   const VertexType& getCellSize() const;

   Index getCellIndex( const Index i ) const;

   void getCellCoordinates( const Index i,
                            CoordinatesType& coordinates ) const;

   Index getVertexIndex( const Index i ) const;

   void getVertexCoordinates( const Index i,
                              CoordinatesType& coordinates ) const;

   /****
    * The type Vertex can have different Real type.
    */
   template< typename Vertex >
   void getCellCenter( const CoordinatesType& coordinates,
                       Vertex& v ) const;

   template< typename Vertex >
   void getVertex( const CoordinatesType& elementCoordinates,
                   Vertex& vertex ) const;

   Index getNumberOfCells() const;

   Index getNumberOfVertices() const;

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceAbsMax( const GridFunction& f1,
                                                        const GridFunction& f2 ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceLpNorm( const GridFunction& f1,
                                                        const GridFunction& f2,
                                                        const typename GridFunction::RealType& p ) const;

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

   tnlStaticVector< 1, IndexType > dimensions;

   tnlStaticVector< 1, RealType > origin;

   IndexType dofs;

};

template< typename Real,
          typename Device,
          typename Index >
class tnlGrid< 2, Real, Device, Index > : public tnlObject
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlStaticVector< 2, Real > VertexType;
   typedef tnlStaticVector< 2, Index > CoordinatesType;
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

   /****
    * The type Vertex can have different Real type.
    */
   template< typename Vertex >
   void getElementCenter( const CoordinatesType& coordinates,
                          Vertex& center ) const;

   Real getElementMeasure( const CoordinatesType& coordinates ) const;

   template< int dx, int dy >
   Real getDualElementMeasure( const CoordinatesType& coordinates ) const;

   template< int dx, int dy >
   void getEdgeNormal( const CoordinatesType& elementCoordinates,
                       VertexType& normal ) const;

   template< int dx, int dy, typename Vertex >
   void getVertex( const CoordinatesType& elementCoordinates,
                   Vertex& vertex ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getAbsMax( const GridFunction& f ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getLpNorm( const GridFunction& f,
                                              const typename GridFunction::RealType& p ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceAbsMax( const GridFunction& f1,
                                                        const GridFunction& f2 ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceLpNorm( const GridFunction& f1,
                                                        const GridFunction& f2,
                                                        const typename GridFunction::RealType& p ) const;

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

   IndexType dofs;

   tnlVector< Real, Device, Index > elementsMeasure, dualElementsMeasure;
   tnlVector< VertexType, Device, Index > edgeNormals, vertices, elementCenters;

};

template< typename Real,
          typename Device,
          typename Index >
class tnlGrid< 3, Real, Device, Index > : public tnlObject
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlStaticVector< 3, Real > VertexType;
   typedef tnlStaticVector< 3, Index > CoordinatesType;
   enum { Dimensions = 3};

   tnlGrid();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   bool setDimensions( const Index xSize, const Index ySize, const Index zSize );

   bool setDimensions( const CoordinatesType& );

   const CoordinatesType& getDimensions() const;

   void setOrigin( const VertexType& origin );

   const VertexType& getOrigin() const;

   void setProportions( const VertexType& proportions );

   const VertexType& getProportions() const;

   void setParametricStep( const VertexType& spaceStep );

   const VertexType& getParametricStep() const;

   Index getElementIndex( const Index i, const Index j, const Index k ) const;

   void getElementCoordinates( const Index i,
                               CoordinatesType& coordinates ) const;

   /****
    * The type Vertex can have different Real type.
    */
   template< typename Vertex >
   void getElementCenter( const CoordinatesType& coordinates,
                          Vertex& center ) const;

   Index getDofs() const;

   template< int dx, int dy, int dz, typename Vertex >
   void getVertex( const CoordinatesType& elementCoordinates,
                   Vertex& vertex ) const;

   Real getElementMeasure( const CoordinatesType& coordinates ) const;

   template< int dx, int dy >
   Real getDualElementMeasure( const CoordinatesType& coordinates ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceAbsMax( const GridFunction& f1,
                                                        const GridFunction& f2 ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceLpNorm( const GridFunction& f1,
                                                        const GridFunction& f2,
                                                        const typename GridFunction::RealType& p ) const;

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

   tnlStaticVector< 3, IndexType > dimensions;

   tnlStaticVector< 3, RealType > origin, proportions;

   IndexType dofs;

};

#include <implementation/mesh/tnlGrid1D_impl.h>
#include <implementation/mesh/tnlGrid2D_impl.h>
#include <implementation/mesh/tnlGrid3D_impl.h>


#endif /* TNLGRID_H_ */
