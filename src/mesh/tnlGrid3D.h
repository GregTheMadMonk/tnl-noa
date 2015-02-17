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

   enum { Dimensions = 3};

   tnlGrid();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   static tnlString getSerializationType();

   virtual tnlString getSerializationTypeVirtual() const;

   void setDimensions( const Index xSize, const Index ySize, const Index zSize );

   void setDimensions( const CoordinatesType& );

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const CoordinatesType& getDimensions() const;

   void setDomain( const VertexType& origin,
                   const VertexType& proportions );
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const VertexType& getOrigin() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const VertexType& getProportions() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const VertexType& getCellProportions() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getCellIndex( const CoordinatesType& cellCoordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   CoordinatesType getCellCoordinates( const IndexType cellIndex ) const;

   template< int nx, int ny, int nz >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getFaceIndex( const CoordinatesType& faceCoordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   CoordinatesType getFaceCoordinates( const Index faceIndex, int& nx, int& ny, int& nz ) const;

   template< int dx, int dy, int dz >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getEdgeIndex( const CoordinatesType& edgeCoordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   CoordinatesType getEdgeCoordinates( const Index edgeIndex, int& dx, int& dy, int& dz ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getVertexIndex( const CoordinatesType& vertexCoordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   CoordinatesType getVertexCoordinates( const Index vertexIndex ) const;

   template< int dx, int dy, int dz >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   IndexType getCellNextToCell( const IndexType& cellIndex ) const;

   template< int nx, int ny, int nz >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   IndexType getFaceNextToCell( const IndexType& cellIndex ) const;

   template< int nx, int ny, int nz >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   IndexType getCellNextToFace( const IndexType& cellIndex ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const RealType& getHx() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const RealType& getHxSquare() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const RealType& getHxInverse() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const RealType& getHxSquareInverse() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const RealType& getHy() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const RealType& getHySquare() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const RealType& getHyInverse() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const RealType& getHySquareInverse() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const RealType& getHz() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const RealType& getHzSquare() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const RealType& getHzInverse() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const RealType& getHzSquareInverse() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const RealType& getHxHy() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const RealType& getHxHz() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const RealType& getHyHz() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const RealType& getHxHyInverse() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const RealType& getHxHzInverse() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const RealType& getHyHzInverse() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   RealType getSmallestSpaceStep() const;

   /****
    * The type Vertex can have different Real type.
    */
#ifdef HAVE_NOT_CXX11
   template< typename Vertex >
#else
   template< typename Vertex = VertexType >
#endif
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Vertex getCellCenter( const CoordinatesType& cellCoordinates ) const;

#ifdef HAVE_NOT_CXX11
   template< typename Vertex >
#else
   template< typename Vertex = VertexType >
#endif
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Vertex getCellCenter( const IndexType& cellIndex ) const;

#ifdef HAVE_NOT_CXX11
   template< int nx, int ny, int nz, typename Vertex >
#else
   template< int nx, int ny, int nz, typename Vertex = VertexType >
#endif
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Vertex getFaceCenter( const CoordinatesType& faceCoordinates ) const;

#ifdef HAVE_NOT_CXX11
   template< int dx, int dy, int dz, typename Vertex >
#else
   template< int dx, int dy, int dz, typename Vertex = VertexType >
#endif
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Vertex getEdgeCenter( const CoordinatesType& edgeCoordinates ) const;

#ifdef HAVE_NOT_CXX11
   template< typename Vertex >
#else
   template< typename Vertex = VertexType >
#endif
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Vertex getVertex( const CoordinatesType& vertexCoordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getNumberOfCells() const;

#ifdef HAVE_NOT_CXX11
   template< int nx,
             int ny,
             int nz >
#else
   template< int nx = 1,
             int ny = 1,
             int nz = 1 >
#endif
#ifdef HAVE_CUDA
   __device__ __host__
#endif
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
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getNumberOfEdges() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getNumberOfVertices() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   bool isBoundaryCell( const CoordinatesType& cellCoordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   bool isBoundaryCell( const IndexType& cellIndex ) const;

   template< int nx, int ny, int nz >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   bool isBoundaryFace( const CoordinatesType& faceCoordinates ) const;

   template< int dx, int dy, int dz >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   bool isBoundaryEdge( const CoordinatesType& edgeCoordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
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



};

#include <mesh/tnlGrid3D_impl.h>

#endif /* SRC_MESH_TNLGRID3D_H_ */
