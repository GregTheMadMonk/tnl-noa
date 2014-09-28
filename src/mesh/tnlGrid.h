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
#include <core/tnlLogger.h>

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

   void setDimensions( const Index xSize );

   void setDimensions( const CoordinatesType& dimensions );

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
   CoordinatesType getCellCoordinates( const Index cellIndex ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getVertexIndex( const CoordinatesType& vertexCoordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   CoordinatesType getVertexCoordinates( const Index vertexCoordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   IndexType getCellXPredecessor( const IndexType& cellIndex ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   IndexType getCellXSuccessor( const IndexType& cellIndex ) const;

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

   /****
    * The type Vertex can have different Real type.
    */
   template< typename Vertex = VertexType >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Vertex getCellCenter( const CoordinatesType& cellCoordinates ) const;

   template< typename Vertex = VertexType >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Vertex getCellCenter( const IndexType& cellIndex ) const;

   template< typename Vertex = VertexType >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Vertex getVertex( const CoordinatesType& vertexCoordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getNumberOfCells() const;

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

#ifdef HAVE_CUDA
   __device__ __host__
#endif
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

   void setDimensions( const Index xSize, const Index ySize );

   void setDimensions( const CoordinatesType& dimensions );

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

   template< int nx, int ny >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getFaceIndex( const CoordinatesType& faceCoordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   CoordinatesType getFaceCoordinates( const Index faceIndex, int& nx, int& ny ) const;


#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getVertexIndex( const CoordinatesType& vertexCoordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   CoordinatesType getVertexCoordinates( const Index vertexIndex ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   IndexType getCellXPredecessor( const IndexType& cellIndex ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   IndexType getCellXSuccessor( const IndexType& cellIndex ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   IndexType getCellYPredecessor( const IndexType& cellIndex ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   IndexType getCellYSuccessor( const IndexType& cellIndex ) const;

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
   const RealType& getHxHy() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const RealType& getHxHyInverse() const;

   /****
    * The type Vertex can have different Real type.
    */
   template< typename Vertex = VertexType >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Vertex getCellCenter( const CoordinatesType& cellCoordinates ) const;

   template< typename Vertex = VertexType >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Vertex getCellCenter( const IndexType& cellIndex ) const;

template< int nx, int ny, typename Vertex = VertexType >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Vertex getFaceCenter( const CoordinatesType& faceCoordinates ) const;

   template< typename Vertex = VertexType >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Vertex getVertex( const CoordinatesType& vertexCoordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getNumberOfCells() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getNumberOfFaces() const;

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

   template< int nx, int ny >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   bool isBoundaryFace( const CoordinatesType& faceCoordinates ) const;

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

   IndexType numberOfCells, numberOfNxFaces, numberOfNyFaces, numberOfFaces, numberOfVertices;

   RealType hx, hxSquare, hxInverse, hxSquareInverse,
            hy, hySquare, hyInverse, hySquareInverse,
            hxhy, hxhyInverse;


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

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   IndexType getCellXPredecessor( const IndexType& cellIndex ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   IndexType getCellXSuccessor( const IndexType& cellIndex ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   IndexType getCellYPredecessor( const IndexType& cellIndex ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   IndexType getCellYSuccessor( const IndexType& cellIndex ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   IndexType getCellZPredecessor( const IndexType& cellIndex ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   IndexType getCellZSuccessor( const IndexType& cellIndex ) const;

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

   /****
    * The type Vertex can have different Real type.
    */
   template< typename Vertex = VertexType >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Vertex getCellCenter( const CoordinatesType& cellCoordinates ) const;

   template< typename Vertex = VertexType >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Vertex getCellCenter( const IndexType& cellIndex ) const;

template< int nx, int ny, int nz, typename Vertex = VertexType >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Vertex getFaceCenter( const CoordinatesType& faceCoordinates ) const;

template< int dx, int dy, int dz, typename Vertex = VertexType >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Vertex getEdgeCenter( const CoordinatesType& edgeCoordinates ) const;

   template< typename Vertex = VertexType >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Vertex getVertex( const CoordinatesType& vertexCoordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getNumberOfCells() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getNumberOfFaces() const;

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

#include <implementation/mesh/tnlGrid1D_impl.h>
#include <implementation/mesh/tnlGrid2D_impl.h>
#include <implementation/mesh/tnlGrid3D_impl.h>


#endif /* TNLGRID_H_ */
