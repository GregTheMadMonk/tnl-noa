/***************************************************************************
                          tnlGrid1D.h  -  description
                             -------------------
    begin                : Feb 13, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
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

#ifndef SRC_MESH_TNLGRID1D_H_
#define SRC_MESH_TNLGRID1D_H_

#include <core/tnlStaticMultiIndex.h>
#include <mesh/grids/tnlGridEntityTopology.h>
#include <mesh/grids/tnlGridEntityCenterGetter.h>

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
   typedef tnlGrid< 1, Real, tnlHost, Index > HostType;
   typedef tnlGrid< 1, Real, tnlCuda, Index > CudaType;
   typedef tnlGrid< 1, Real, Device, Index > ThisType;
   
   template< int i > using EntityOrientation = tnlStaticMultiIndex1D< i >;
   
   typedef tnlGridEntityTopology< ThisType, 1, EntityOrientation<  0 > > Cell;
   typedef tnlGridEntityTopology< ThisType, 0, EntityOrientation<  0 > > Vertex;
   
   enum { Dimensions = 1};

   tnlGrid();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   static tnlString getSerializationType();

   virtual tnlString getSerializationTypeVirtual() const;

   void setDimensions( const Index xSize );

   void setDimensions( const CoordinatesType& dimensions );

   __cuda_callable__
   const CoordinatesType& getDimensions() const;

   void setDomain( const VertexType& origin,
                   const VertexType& proportions );

   __cuda_callable__
   const VertexType& getOrigin() const;

   __cuda_callable__
   const VertexType& getProportions() const;

   __cuda_callable__
   const VertexType& getCellProportions() const;

   __cuda_callable__
   Index getCellIndex( const CoordinatesType& cellCoordinates ) const;

   __cuda_callable__
   CoordinatesType getCellCoordinates( const Index cellIndex ) const;

   __cuda_callable__
   Index getVertexIndex( const CoordinatesType& vertexCoordinates ) const;

   __cuda_callable__
   CoordinatesType getVertexCoordinates( const Index vertexCoordinates ) const;

   template< int dx >
   __cuda_callable__
   IndexType getCellNextToCell( const IndexType& cellIndex ) const;

   __cuda_callable__
   const RealType& getHx() const;

   __cuda_callable__
   const RealType& getHxSquare() const;

   __cuda_callable__
   const RealType& getHxInverse() const;

   __cuda_callable__
   const RealType& getHxSquareInverse() const;

   __cuda_callable__
   RealType getSmallestSpaceStep() const;

   /****
    * The type Vertex can have different Real type.
    */
#ifdef HAVE_NOT_CXX11
   template< typename EntityTopology, 
             typename Vertex >
#else
   template< typename EntityTopology,
             typename Vertex = VertexType >
#endif
   __cuda_callable__
   Vertex getEntityCenter( const CoordinatesType& cellCoordinates ) const;

#ifdef HAVE_NOT_CXX11
   template< typename EntityTopology, 
             typename Vertex >
#else
   template< typename EntityTopology,
             typename Vertex = VertexType >
#endif
   __cuda_callable__
   Vertex getEntityCenter( const IndexType& cellIndex ) const;
   
#ifdef HAVE_NOT_CXX11
   template< typename Vertex >
#else
   template< typename Vertex = VertexType >
#endif
   __cuda_callable__
   Vertex getCellCenter( const CoordinatesType& cellCoordinates ) const;

#ifdef HAVE_NOT_CXX11
   template< typename Vertex >
#else
   template< typename Vertex = VertexType >
#endif
   __cuda_callable__
   Vertex getCellCenter( const IndexType& cellIndex ) const;

#ifdef HAVE_NOT_CXX11
   template< typename Vertex >
#else
   template< typename Vertex = VertexType >
#endif
   __cuda_callable__
   Vertex getVertex( const CoordinatesType& vertexCoordinates ) const;

   __cuda_callable__
   Index getNumberOfCells() const;

   __cuda_callable__
   Index getNumberOfVertices() const;

   __cuda_callable__
   bool isBoundaryCell( const CoordinatesType& cellCoordinates ) const;

   __cuda_callable__
   bool isBoundaryCell( const IndexType& cellIndex ) const;

   __cuda_callable__
   bool isBoundaryVertex( const CoordinatesType& vertexCoordinates ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceAbsMax( const GridFunction& f1,
                                                        const GridFunction& f2 ) const;

   template< typename GridFunction >
   typename GridFunction::RealType getDifferenceLpNorm( const GridFunction& f1,
                                                        const GridFunction& f2,
                                                        const typename GridFunction::RealType& p ) const;

   /****
    *  Method for saving the object to a file as a binary data
    */
   bool save( tnlFile& file ) const;

   /****
    *  Method for restoring the object from a file
    */
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   bool writeMesh( const tnlString& fileName,
                   const tnlString& format ) const;

   template< typename MeshFunction >
   bool write( const MeshFunction& function,
               const tnlString& fileName,
               const tnlString& format ) const;

   void writeProlog( tnlLogger& logger );

   protected:

   void computeSpaceSteps();

   CoordinatesType dimensions;

   VertexType origin, proportions, cellProportions;

   IndexType numberOfCells, numberOfVertices;

   RealType hx, hxSquare, hxInverse, hxSquareInverse;

};

#include <mesh/grids/tnlGrid1D_impl.h>

#endif /* SRC_MESH_TNLGRID1D_H_ */
