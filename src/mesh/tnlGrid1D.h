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

   enum { Dimensions = 1};

   tnlGrid();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   static tnlString getSerializationType();

   virtual tnlString getSerializationTypeVirtual() const;

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

   template< int dx >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   IndexType getCellNextToCell( const IndexType& cellIndex ) const;

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

#include <mesh/tnlGrid1D_impl.h>

#endif /* SRC_MESH_TNLGRID1D_H_ */
