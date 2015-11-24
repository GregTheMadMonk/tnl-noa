/***************************************************************************
                          tnlGrid3D.h  -  description
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

#ifndef SRC_MESH_TNLGRID3D_H_
#define SRC_MESH_TNLGRID3D_H_

#include <core/tnlStaticMultiIndex.h>
#include <mesh/grids/tnlGridEntityTopology.h>
#include <mesh/grids/tnlGridEntityGetter.h>
#include <mesh/grids/tnlNeighbourGridEntityGetter.h>

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
   typedef tnlGrid< 3, Real, tnlHost, Index > HostType;
   typedef tnlGrid< 3, Real, tnlCuda, Index > CudaType;

   typedef tnlGrid< 3, Real, Device, Index > ThisType;

   template< int EntityDimensions > using GridEntity = 
      tnlGridEntity< ThisType, EntityDimensions >;   
   
   enum { Dimensions = 3 };
   enum { Cells = 3 };
   enum { Faces = 2 };
   enum { Edges = 1 };
   enum { Vertices = 0 };

   tnlGrid();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   static tnlString getSerializationType();

   virtual tnlString getSerializationTypeVirtual() const;

   void setDimensions( const Index xSize, const Index ySize, const Index zSize );

   void setDimensions( const CoordinatesType& );

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
   
   template< int EntityDimensions >
   IndexType getEntitiesCount() const;
   
   template< int EntityDimensions >
   __cuda_callable__
   GridEntity< EntityDimensions > getEntity( const IndexType& entityIndex ) const;
   
   template< int EntityDimensions >
   __cuda_callable__
   Index getEntityIndex( const GridEntity< EntityDimensions >& entity ) const;

   /****
    * The type Vertex can have different Real type.
    */
#ifdef HAVE_NOT_CXX11
   template< int EntityDimensions, 
             typename Vertex >
#else
   template< int EntityDimensions,
             typename Vertex = VertexType >
#endif
   __cuda_callable__
   Vertex getEntityCenter( const GridEntity< EntityDimensions >& entity ) const;
   
   
   
   template< int dx, int dy, int dz >
   __cuda_callable__
   IndexType getCellNextToCell( const IndexType& cellIndex ) const;

   template< int nx, int ny, int nz >
   __cuda_callable__
   IndexType getFaceNextToCell( const IndexType& cellIndex ) const;

   template< int nx, int ny, int nz >
   __cuda_callable__
   IndexType getCellNextToFace( const IndexType& cellIndex ) const;

   __cuda_callable__
   const RealType& getHx() const;

   __cuda_callable__
   const RealType& getHxSquare() const;

   __cuda_callable__
   const RealType& getHxInverse() const;

   __cuda_callable__
   const RealType& getHxSquareInverse() const;

   __cuda_callable__
   const RealType& getHy() const;

   __cuda_callable__
   const RealType& getHySquare() const;

   __cuda_callable__
   const RealType& getHyInverse() const;

   __cuda_callable__
   const RealType& getHySquareInverse() const;

   __cuda_callable__
   const RealType& getHz() const;

   __cuda_callable__
   const RealType& getHzSquare() const;

   __cuda_callable__
   const RealType& getHzInverse() const;

   __cuda_callable__
   const RealType& getHzSquareInverse() const;

   __cuda_callable__
   const RealType& getHxHy() const;

   __cuda_callable__
   const RealType& getHxHz() const;

   __cuda_callable__
   const RealType& getHyHz() const;

   __cuda_callable__
   const RealType& getHxHyInverse() const;

   __cuda_callable__
   const RealType& getHxHzInverse() const;

   __cuda_callable__
   const RealType& getHyHzInverse() const;

   __cuda_callable__
   RealType getSmallestSpaceStep() const;

#ifdef HAVE_NOT_CXX11
   template< int nx,
             int ny,
             int nz >
#else
   template< int nx = 1,
             int ny = 1,
             int nz = 1 >
#endif
   __cuda_callable__
   Index getNumberOfFaces() const;

#ifdef HAVE_NOT_CXX11
   template< int dx,
             int dy,
             int dz >
#else
   template< int dx = 1,
             int dy = 1,
             int dz = 1 >
#endif
   __cuda_callable__
   Index getNumberOfEdges() const;

   __cuda_callable__
   Index getNumberOfVertices() const;

   __cuda_callable__
   bool isBoundaryCell( const CoordinatesType& cellCoordinates ) const;

   __cuda_callable__
   bool isBoundaryCell( const IndexType& cellIndex ) const;

   template< int nx, int ny, int nz >
   __cuda_callable__
   bool isBoundaryFace( const CoordinatesType& faceCoordinates ) const;

   template< int dx, int dy, int dz >
   __cuda_callable__
   bool isBoundaryEdge( const CoordinatesType& edgeCoordinates ) const;

   __cuda_callable__
   bool isBoundaryVertex( const CoordinatesType& vertexCoordinates ) const;

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

   void writeProlog( tnlLogger& logger );

   protected:

   void computeSpaceSteps();

   CoordinatesType dimensions;

   VertexType origin, proportions, cellProportions;

   IndexType numberOfCells,
             numberOfNxFaces, numberOfNyFaces, numberOfNzFaces, numberOfNxAndNyFaces, numberOfFaces,
             numberOfDxEdges, numberOfDyEdges, numberOfDzEdges, numberOfDxAndDyEdges, numberOfEdges,
             numberOfVertices;
   IndexType cellZNeighboursStep;

   RealType hx, hxSquare, hxInverse, hxSquareInverse,
            hy, hySquare, hyInverse, hySquareInverse,
            hz, hzSquare, hzInverse, hzSquareInverse,
            hxhy, hxhz, hyhz,
            hxhyInverse, hxhzInverse, hyhzInverse;

   friend class tnlGridEntityGetter< ThisType, 3 >;
   friend class tnlGridEntityGetter< ThisType, 2 >;
   friend class tnlGridEntityGetter< ThisType, 1 >;
   friend class tnlGridEntityGetter< ThisType, 0 >;
};

#include <mesh/grids/tnlGrid3D_impl.h>

#endif /* SRC_MESH_TNLGRID3D_H_ */
