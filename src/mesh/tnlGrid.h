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
   Index getCellIndex( const CoordinatesType& coordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   CoordinatesType getCellCoordinates( const Index i ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getVertexIndex( const CoordinatesType& coordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   CoordinatesType getVertexCoordinates( const Index i ) const;

   /****
    * The type Vertex can have different Real type.
    */
   template< typename Vertex >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Vertex getCellCenter( const CoordinatesType& coordinates ) const;

   template< typename Vertex >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Vertex getVertex( const CoordinatesType& elementCoordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getNumberOfCells() const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getNumberOfVertices() const;

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

   protected:

   CoordinatesType dimensions;

   VertexType origin, proportions, cellProportions;

   IndexType numberOfCells, numberOfVertices;

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
   Index getCellIndex( const CoordinatesType& coordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   CoordinatesType getCellCoordinates( const Index i ) const;

   template< int nx, int ny >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getFaceIndex( const CoordinatesType& coordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   CoordinatesType getFaceCoordinates( const Index i, int& nx, int& ny ) const;


#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getVertexIndex( const CoordinatesType& coordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   CoordinatesType getVertexCoordinates( const Index i ) const;

   /****
    * The type Vertex can have different Real type.
    */
   template< typename Vertex >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Vertex getCellCenter( const CoordinatesType& coordinates ) const;

template< int nx, int ny >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getFaceCenter( const Index i,
                        CoordinatesType& coordinates ) const;

   template< typename Vertex >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Vertex getVertex( const CoordinatesType& elementCoordinates ) const;

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

   VertexType origin, proportions, cellProportions;

   IndexType numberOfCells, numberOfNxFaces, numberOfFaces, numberOfVertices;


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
   Index getCellIndex( const CoordinatesType& coordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   CoordinatesType getCellCoordinates( const Index i ) const;

   template< int nx, int ny, int nz >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getFaceIndex( const CoordinatesType& coordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   CoordinatesType getFaceCoordinates( const Index i, int& nx, int& ny, int& nz ) const;

   template< int dx, int dy, int dz >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getEdgeIndex( const CoordinatesType& coordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   CoordinatesType getEdgeCoordinates( const Index i, int& dx, int& dy, int& dz ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getVertexIndex( const CoordinatesType& coordinates ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   CoordinatesType getVertexCoordinates( const Index i ) const;

   /****
    * The type Vertex can have different Real type.
    */
   template< typename Vertex >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Vertex getCellCenter( const CoordinatesType& coordinates ) const;

template< int nx, int ny >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getFaceCenter( const Index i,
                        CoordinatesType& coordinates ) const;

   template< typename Vertex >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Vertex getVertex( const CoordinatesType& elementCoordinates ) const;

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

   VertexType origin, proportions, cellProportions;

   IndexType numberOfCells, numberOfFaces, numberOfEdges, numberOfVertices;

};

#include <implementation/mesh/tnlGrid1D_impl.h>
#include <implementation/mesh/tnlGrid2D_impl.h>
#include <implementation/mesh/tnlGrid3D_impl.h>


#endif /* TNLGRID_H_ */
