/***************************************************************************
                          tnlGrid2D.h  -  description
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

#ifndef SRC_MESH_TNLGRID2D_H_
#define SRC_MESH_TNLGRID2D_H_

#include <core/tnlStaticMultiIndex.h>
#include <mesh/grids/tnlGridEntityTopology.h>
#include <mesh/grids/tnlGridEntityGetter.h>

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
   typedef tnlGrid< 2, Real, tnlHost, Index > HostType;
   typedef tnlGrid< 2, Real, tnlCuda, Index > CudaType;
   
   typedef tnlGrid< 2, Real, Device, Index > ThisType;
   template< int i1, int i2 > using EntityOrientation = tnlStaticMultiIndex2D< i1, i2 >;
   template< int i1, int i2 > using EntityProportions = tnlStaticMultiIndex2D< i1, i2 >;   
   
   typedef tnlGridEntityTopology< ThisType,
                                  2,
                                  EntityOrientation< 0, 0 >,
                                  EntityProportions< 1, 1 > > Cell;
   
   /****
    * ( n1, n2 ) is a face outer normal. If both are zeros it means any face.
    */
   template< int n1 = 0, int n2 = 0 > using Face = 
      tnlGridEntityTopology< ThisType,
                             1,
                             EntityOrientation<  n1,  n2 >,
                             EntityProportions< 1 - tnlConstAbs( n1 ),
                                                1 - tnlConstAbs( n2 ) > >;
   
   typedef tnlGridEntityTopology< ThisType,
                                  0,
                                  EntityOrientation< 0, 0 >,
                                  EntityProportions< 1, 1 > > Vertex;

   template< int EntityDimensions > using GridEntity = 
      tnlGridEntity< ThisType, EntityDimensions >;   

   enum { Dimensions = 2};

   tnlGrid();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   static tnlString getSerializationType();

   virtual tnlString getSerializationTypeVirtual() const;

   void setDimensions( const Index xSize, const Index ySize );

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
   

   
   
   
   
   
   



   __cuda_callable__
   Index getCellIndex( const CoordinatesType& cellCoordinates ) const;

   __cuda_callable__
   CoordinatesType getCellCoordinates( const IndexType cellIndex ) const;

   template< int nx, int ny >
   __cuda_callable__
   Index getFaceIndex( const CoordinatesType& faceCoordinates ) const;

   __cuda_callable__
   CoordinatesType getFaceCoordinates( const Index faceIndex, int& nx, int& ny ) const;


   __cuda_callable__
   Index getVertexIndex( const CoordinatesType& vertexCoordinates ) const;

   __cuda_callable__
   CoordinatesType getVertexCoordinates( const Index vertexIndex ) const;

   template< int dx, int dy >
   __cuda_callable__
   IndexType getCellNextToCell( const IndexType& cellIndex ) const;

   template< int nx, int ny >
   __cuda_callable__
   IndexType getFaceNextToCell( const IndexType& cellIndex ) const;

   template< int nx, int ny >
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
   const RealType& getHxHy() const;

   __cuda_callable__
   const RealType& getHxHyInverse() const;

   __cuda_callable__
   RealType getSmallestSpaceStep() const;


   /****
    * The type Vertex can have different Real type.
    */

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
   template< int nx, int ny, typename Vertex >
#else
   template< int nx, int ny, typename Vertex = VertexType >
#endif
   __cuda_callable__
   Vertex getFaceCenter( const CoordinatesType& faceCoordinates ) const;

#ifdef HAVE_NOT_CXX11
   template< typename Vertex >
#else
   template< typename Vertex = VertexType >
#endif
   __cuda_callable__
   Vertex getVertex( const CoordinatesType& vertexCoordinates ) const;

   __cuda_callable__
   Index getNumberOfCells() const;

#ifdef HAVE_NOT_CXX11
   template< int nx,
             int ny >
#else
   template< int nx = 1,
             int ny = 1 >
#endif
   __cuda_callable__
   Index getNumberOfFaces() const;

   __cuda_callable__
   Index getNumberOfVertices() const;

   __cuda_callable__
   bool isBoundaryCell( const CoordinatesType& cellCoordinates ) const;

   __cuda_callable__
   bool isBoundaryCell( const IndexType& cellIndex ) const;

   template< int nx, int ny >
   __cuda_callable__
   bool isBoundaryFace( const CoordinatesType& faceCoordinates ) const;

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

   __cuda_callable__
   void computeSpaceSteps();

   CoordinatesType dimensions;

   VertexType origin, proportions, cellProportions;

   IndexType numberOfCells, numberOfNxFaces, numberOfNyFaces, numberOfFaces, numberOfVertices;

   RealType hx, hxSquare, hxInverse, hxSquareInverse,
            hy, hySquare, hyInverse, hySquareInverse,
            hxhy, hxhyInverse;


};

#include <mesh/grids/tnlGrid2D_impl.h>

#endif /* SRC_MESH_TNLGRID2D_H_ */
