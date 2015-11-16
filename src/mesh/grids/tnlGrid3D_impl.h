/***************************************************************************
                          tnlGrid3D_impl.h  -  description
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

#ifndef TNLGRID3D_IMPL_H_
#define TNLGRID3D_IMPL_H_

#include <iomanip>
#include <core/tnlAssert.h>
#include <mesh/grids/tnlGridEntityCenterGetter_impl.h>
#include <mesh/grids/tnlGrid3D.h>

template< typename Real,
          typename Device,
          typename Index >
tnlGrid< 3, Real, Device, Index > :: tnlGrid()
: numberOfCells( 0 ),
  numberOfNxFaces( 0 ),
  numberOfNyFaces( 0 ),
  numberOfNzFaces( 0 ),
  numberOfNxAndNyFaces( 0 ),
  numberOfFaces( 0 ),
  numberOfDxEdges( 0 ),
  numberOfDyEdges( 0 ),
  numberOfDzEdges( 0 ),
  numberOfDxAndDyEdges( 0 ),
  numberOfEdges( 0 ),
  numberOfVertices( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlGrid< 3, Real, Device, Index > :: getType()
{
   return tnlString( "tnlGrid< " ) +
          tnlString( Dimensions ) + ", " +
          tnlString( ::getType< RealType >() ) + ", " +
          tnlString( Device :: getDeviceType() ) + ", " +
          tnlString( ::getType< IndexType >() ) + " >";
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlGrid< 3, Real, Device, Index > :: getTypeVirtual() const
{
   return this -> getType();
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlGrid< 3, Real, Device, Index > :: getSerializationType()
{
   return HostType::getType();
};

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlGrid< 3, Real, Device, Index > :: getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 3, Real, Device, Index > :: computeSpaceSteps()
{
   if( this->getDimensions().x() > 0 &&
       this->getDimensions().y() > 0 &&
       this->getDimensions().z() > 0 )
   {
      this->cellProportions.x() = this->proportions.x() / ( Real ) this->getDimensions().x();
      this->cellProportions.y() = this->proportions.y() / ( Real ) this->getDimensions().y();
      this->cellProportions.z() = this->proportions.z() / ( Real ) this->getDimensions().z();
      this->hx = this->proportions.x() / ( Real ) this->getDimensions().x();
      this->hxSquare = this->hx * this->hx;
      this->hxInverse = 1.0 / this->hx;
      this->hxSquareInverse = this->hxInverse * this->hxInverse;
      this->hy = this->proportions.y() / ( Real ) this->getDimensions().y();
      this->hySquare = this->hy * this->hy;
      this->hyInverse = 1.0 / this->hy;
      this->hySquareInverse = this->hyInverse * this->hyInverse;
      this->hz = this->proportions.z() / ( Real ) this->getDimensions().z();
      this->hzSquare = this->hz * this->hz;
      this->hzInverse = 1.0 / this->hz;
      this->hzSquareInverse = this->hzInverse * this->hzInverse;
      this->hxhy = this->hx * this->hy;
      this->hxhz = this->hx * this->hz;
      this->hyhz = this->hy * this->hz;
      this->hxhyInverse = 1.0 / this->hxhy;
      this->hxhzInverse = 1.0 / this->hxhz;
      this->hyhzInverse = 1.0 / this->hyhz;
   }
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 3, Real, Device, Index > :: setDimensions( const Index xSize, const Index ySize, const Index zSize )
{
   tnlAssert( xSize > 0, cerr << "xSize = " << xSize );
   tnlAssert( ySize > 0, cerr << "ySize = " << ySize );
   tnlAssert( zSize > 0, cerr << "zSize = " << zSize );

   this->dimensions.x() = xSize;
   this->dimensions.y() = ySize;
   this->dimensions.z() = zSize;
   this->numberOfCells = xSize * ySize * zSize;
   this->numberOfNxFaces = ( xSize + 1 ) * ySize * zSize;
   this->numberOfNyFaces = xSize * ( ySize + 1 ) * zSize;
   this->numberOfNzFaces = xSize * ySize * ( zSize + 1 );
   this->numberOfNxAndNyFaces = this->numberOfNxFaces + this->numberOfNyFaces;
   this->numberOfFaces = this->numberOfNxFaces +
                         this->numberOfNyFaces +
                         this->numberOfNzFaces;
   this->numberOfDxEdges = xSize * ( ySize + 1 ) * ( zSize + 1 );
   this->numberOfDyEdges = ( xSize + 1 ) * ySize * ( zSize + 1 );
   this->numberOfDzEdges = ( xSize + 1 ) * ( ySize + 1 ) * zSize;
   this->numberOfDxAndDyEdges = this->numberOfDxEdges + this->numberOfDyEdges;
   this->numberOfEdges = this->numberOfDxEdges +
                         this->numberOfDyEdges +
                         this->numberOfDzEdges;
   this->numberOfVertices = ( xSize + 1 ) * ( ySize + 1 ) * ( zSize + 1 );
   
   this->cellZNeighboursStep = xSize * ySize;

   computeSpaceSteps();
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 3, Real, Device, Index > :: setDimensions( const CoordinatesType& dimensions )
{
   return this -> setDimensions( dimensions. x(), dimensions. y(), dimensions. z() );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const typename tnlGrid< 3, Real, Device, Index > :: CoordinatesType&
   tnlGrid< 3, Real, Device, Index > :: getDimensions() const
{
   return this -> dimensions;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 3, Real, Device, Index > :: setDomain( const VertexType& origin,
                                                     const VertexType& proportions )
{
   this->origin = origin;
   this->proportions = proportions;
   computeSpaceSteps();
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const typename tnlGrid< 3, Real, Device, Index >::VertexType&
tnlGrid< 3, Real, Device, Index >::getOrigin() const
{
   return this->origin;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const typename tnlGrid< 3, Real, Device, Index > :: VertexType&
   tnlGrid< 3, Real, Device, Index > :: getProportions() const
{
	return this->proportions;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const typename tnlGrid< 3, Real, Device, Index > :: VertexType&
   tnlGrid< 3, Real, Device, Index > :: getCellProportions() const
{
   return this->cellProportions;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlGrid< 3, Real, Device, Index > :: getCellIndex( const CoordinatesType& cellCoordinates ) const
{
   tnlAssert( cellCoordinates.x() >= 0 && cellCoordinates.x() < this->getDimensions().x(),
              cerr << "cellCoordinates.x() = " << cellCoordinates.x()
                   << " this->getDimensions().x() = " << this->getDimensions().x() );
   tnlAssert( cellCoordinates.y() >= 0 && cellCoordinates.y() < this->getDimensions().y(),
              cerr << "cellCoordinates.y() = " << cellCoordinates.y()
                   << " this->getDimensions().y() = " << this->getDimensions().y() );
   tnlAssert( cellCoordinates.z() >= 0 && cellCoordinates.z() < this->getDimensions().z(),
              cerr << "cellCoordinates.z() = " << cellCoordinates.z()
                   << " this->getDimensions().z() = " << this->getDimensions().z() );

   return ( cellCoordinates.z() * this->dimensions.y() + cellCoordinates.y() ) * this->dimensions.x() + cellCoordinates.x();
}

template< typename Real,
          typename Device,
          typename Index >
typename tnlGrid< 3, Real, Device, Index >::CoordinatesType
tnlGrid< 3, Real, Device, Index > :: getCellCoordinates( const Index cellIndex ) const
{
   tnlAssert( cellIndex >= 0 && cellIndex < this->getNumberOfCells(),
              cerr << " cellIndex = " << cellIndex
                   << " this->getNumberOfCells() = " << this->getNumberOfCells() );

   return CoordinatesType( cellIndex % this->dimensions.x(),
                           ( cellIndex / this->dimensions.x() ) % this->dimensions.y(),
                           cellIndex / ( this->dimensions.x() * this->dimensions.y() ) );
}

template< typename Real,
          typename Device,
          typename Index >
template< int nx, int ny, int nz >
__cuda_callable__
Index tnlGrid< 3, Real, Device, Index >::getFaceIndex( const CoordinatesType& faceCoordinates ) const
{
   tnlStaticAssert( nx >= 0 && ny >= 0 && nz >= 0 && nx + ny + nz == 1, "Wrong template parameters nx or ny or nz." );
   if( nx )
   {
      tnlAssert( faceCoordinates.x() >= 0 && faceCoordinates.x() < this->getDimensions().x() + 1,
                 cerr << "faceCoordinates.x() = " << faceCoordinates.x()
                      << " this->getDimensions().x() + 1 = " << this->getDimensions().x() + 1 );
      tnlAssert( faceCoordinates.y() >= 0 && faceCoordinates.y() < this->getDimensions().y(),
                 cerr << "faceCoordinates.y() = " << faceCoordinates.y()
                      << " this->getDimensions().y() = " << this->getDimensions().y() );
      tnlAssert( faceCoordinates.z() >= 0 && faceCoordinates.z() < this->getDimensions().z(),
                 cerr << "faceCoordinates.z() = " << faceCoordinates.z()
                      << " this->getDimensions().z() = " << this->getDimensions().z() );
      return ( faceCoordinates.z() * this->getDimensions().y() + faceCoordinates.y() ) * ( this->getDimensions().x() + 1 ) + faceCoordinates.x();
   }
   if( ny )
   {
      tnlAssert( faceCoordinates.x() >= 0 && faceCoordinates.x() < this->getDimensions().x(),
                 cerr << "faceCoordinates.x() = " << faceCoordinates.x()
                      << " this->getDimensions().x() = " << this->getDimensions().x() );
      tnlAssert( faceCoordinates.y() >= 0 && faceCoordinates.y() < this->getDimensions().y() + 1,
                 cerr << "faceCoordinates.y() = " << faceCoordinates.y()
                      << " this->getDimensions().y() + 1 = " << this->getDimensions().y() + 1 );
      tnlAssert( faceCoordinates.z() >= 0 && faceCoordinates.z() < this->getDimensions().z(),
                 cerr << "faceCoordinates.z() = " << faceCoordinates.z()
                      << " this->getDimensions().z() = " << this->getDimensions().z() );

      return this->numberOfNxFaces + ( faceCoordinates.z() * ( this->getDimensions().y() + 1 ) + faceCoordinates.y() ) * this->getDimensions().x() + faceCoordinates.x();
   }
   if( nz )
   {
      tnlAssert( faceCoordinates.x() >= 0 && faceCoordinates.x() < this->getDimensions().x(),
                 cerr << "faceCoordinates.x() = " << faceCoordinates.x()
                      << " this->getDimensions().x() = " << this->getDimensions().x() );
      tnlAssert( faceCoordinates.y() >= 0 && faceCoordinates.y() < this->getDimensions().y(),
                 cerr << "faceCoordinates.y() = " << faceCoordinates.y()
                      << " this->getDimensions().y()= " << this->getDimensions().y() );
      tnlAssert( faceCoordinates.z() >= 0 && faceCoordinates.z() < this->getDimensions().z() + 1,
                 cerr << "faceCoordinates.z() = " << faceCoordinates.z()
                      << " this->getDimensions().z() + 1 = " << this->getDimensions().z() + 1 );

      return this->numberOfNxAndNyFaces + ( faceCoordinates.z() * this->getDimensions().y() + faceCoordinates.y() ) * this->getDimensions().x() + faceCoordinates.x();
   }
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
typename tnlGrid< 3, Real, Device, Index >::CoordinatesType
tnlGrid< 3, Real, Device, Index >::getFaceCoordinates( const Index faceIndex, int& nx, int& ny, int& nz ) const
{
   tnlAssert( faceIndex >= 0 && faceIndex < this->getNumberOfFaces(),
              cerr << " faceIndex = " << faceIndex
                   << " this->getNumberOfFaces() = " << this->getNumberOfFaces() );
   if( faceIndex < this->numberOfNxFaces )
   {
      nx = 1;
      ny = 0;
      nz = 0;
      const IndexType aux = this->getDimensions().x() + 1;
      return CoordinatesType( faceIndex % aux,
                             ( faceIndex / aux ) % this->getDimensions().y(),
                             faceIndex / ( aux * this->getDimensions().y() ) );
   }
   if( faceIndex < this->numberOfNxAndNyFaces )
   {
      nx = 0;
      ny = 1;
      nz = 0;
      const IndexType i = faceIndex - this->numberOfNxFaces;
      const IndexType aux = this->getDimensions().y() + 1;
      return CoordinatesType( i % this->getDimensions().x(),
                              ( i / this->getDimensions().x() ) % aux,
                              i / ( aux * this->getDimensions().x() ) );
   }
   nx = 0;
   ny = 0;
   nz = 1;
   const IndexType i = faceIndex - this->numberOfNxAndNyFaces;
   return CoordinatesType( i % this->getDimensions().x(),
                           ( i / this->getDimensions().x() ) % this->getDimensions().y(),
                           i / ( this->getDimensions().x() * this->getDimensions().y() ) );
}

template< typename Real,
          typename Device,
          typename Index >
template< int dx, int dy, int dz >
__cuda_callable__
Index tnlGrid< 3, Real, Device, Index > :: getEdgeIndex( const CoordinatesType& edgeCoordinates ) const
{
   tnlStaticAssert( dx >= 0 && dy >= 0 && dz >= 0 && dx + dy + dz == 1, "Wrong template parameters dx or dy or dz.");
   if( dx )
   {
      tnlAssert( edgeCoordinates.x() >= 0 && edgeCoordinates.x() < this->getDimensions().x(),
                 cerr << "edgeCoordinates.x() = " << edgeCoordinates.x()
                      << " this->getDimensions().x() = " << this->getDimensions().x() );
      tnlAssert( edgeCoordinates.y() >= 0 && edgeCoordinates.y() < this->getDimensions().y() + 1,
                 cerr << "edgeCoordinates.y() = " << edgeCoordinates.y()
                      << " this->getDimensions().y() + 1 = " << this->getDimensions().y() + 1 );
      tnlAssert( edgeCoordinates.z() >= 0 && edgeCoordinates.z() < this->getDimensions().z() + 1,
                 cerr << "edgeCoordinates.z() = " << edgeCoordinates.z()
                      << " this->getDimensions().z() + 1 = " << this->getDimensions().z() + 1 );
      return ( edgeCoordinates.z() * ( this->getDimensions().y() + 1 ) + edgeCoordinates.y() ) * this->getDimensions().x() + edgeCoordinates.x();
   }
   if( dy )
   {
      tnlAssert( edgeCoordinates.x() >= 0 && edgeCoordinates.x() < this->getDimensions().x() + 1,
                 cerr << "edgeCoordinates.x() = " << edgeCoordinates.x()
                      << " this->getDimensions().x() + 1 = " << this->getDimensions().x() + 1 );
      tnlAssert( edgeCoordinates.y() >= 0 && edgeCoordinates.y() < this->getDimensions().y(),
                 cerr << "edgeCoordinates.y() = " << edgeCoordinates.y()
                      << " this->getDimensions().y() = " << this->getDimensions().y() );
      tnlAssert( edgeCoordinates.z() >= 0 && edgeCoordinates.z() < this->getDimensions().z() + 1,
                 cerr << "edgeCoordinates.z() = " << edgeCoordinates.z()
                      << " this->getDimensions().z() + 1 = " << this->getDimensions().z() + 1; );

      return this->numberOfDxEdges + ( edgeCoordinates.z() * this->getDimensions().y() + edgeCoordinates.y() ) * ( this->getDimensions().x() + 1 ) + edgeCoordinates.x();
   }
   if( dz )
   {
      tnlAssert( edgeCoordinates.x() >= 0 && edgeCoordinates.x() < this->getDimensions().x() + 1,
                 cerr << "edgeCoordinates.x() = " << edgeCoordinates.x()
                      << " this->getDimensions().x() = " << this->getDimensions().x() );
      tnlAssert( edgeCoordinates.y() >= 0 && edgeCoordinates.y() < this->getDimensions().y() + 1,
                 cerr << "edgeCoordinates.y() = " << edgeCoordinates.y()
                      << " this->getDimensions().y() + 1 = " << this->getDimensions().y() + 1 );
      tnlAssert( edgeCoordinates.z() >= 0 && edgeCoordinates.z() < this->getDimensions().z(),
                 cerr << "edgeCoordinates.z() = " << edgeCoordinates.z()
                      << " this->getDimensions().z() = " << this->getDimensions().z() );

      return this->numberOfDxAndDyEdges + ( edgeCoordinates.z() * ( this->getDimensions().y() + 1 ) + edgeCoordinates.y() ) * ( this->getDimensions().x() + 1 ) + edgeCoordinates.x();
   }
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
typename tnlGrid< 3, Real, Device, Index > :: CoordinatesType
tnlGrid< 3, Real, Device, Index > :: getEdgeCoordinates( const Index edgeIndex, int& dx, int& dy, int& dz ) const
{
   tnlAssert( edgeIndex >= 0 && edgeIndex < this->getNumberOfEdges(),
              cerr << " edgeIndex = " << edgeIndex
                   << " this->getNumberOfEdges() = " << this->getNumberOfEdges() );
   if( edgeIndex < this->numberOfDxEdges )
   {
      dx = 1;
      dy = 0;
      dz = 0;
      const IndexType aux = this->getDimensions().y() + 1;
      return CoordinatesType( edgeIndex % this->getDimensions().x(),
                             ( edgeIndex / this->getDimensions().x() ) % aux,
                             edgeIndex / ( this->getDimensions().x() * aux ) );
   }
   if( edgeIndex < this->numberOfDxAndDyEdges )
   {
      dx = 0;
      dy = 1;
      dz = 0;
      const IndexType i = edgeIndex - this->numberOfDxEdges;
      const IndexType aux = this->getDimensions().x() + 1;
      return CoordinatesType( i % aux,
                              ( i / aux ) % this->getDimensions().y(),
                              i / ( aux * this->getDimensions().y() ) );
   }
   dx = 0;
   dy = 0;
   dz = 1;
   const IndexType i = edgeIndex - this->numberOfDxAndDyEdges;
   const IndexType aux1 = this->getDimensions().x() + 1;
   const IndexType aux2 = this->getDimensions().y() + 1;
   return CoordinatesType( i % aux1,
                           ( i / aux1 ) % aux2,
                           i / ( aux1 * aux2 ) );

}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index tnlGrid< 3, Real, Device, Index > :: getVertexIndex( const CoordinatesType& vertexCoordinates ) const
{
   tnlAssert( vertexCoordinates.x() >= 0 && vertexCoordinates.x() < this->getDimensions().x() + 1,
              cerr << "vertexCoordinates.x() = " << vertexCoordinates.x()
                   << " this->getDimensions().x() + 1 = " << this->getDimensions().x() + 1 );
   tnlAssert( vertexCoordinates.y() >= 0 && vertexCoordinates.y() < this->getDimensions().y() + 1,
              cerr << "vertexCoordinates.y() = " << vertexCoordinates.y()
                   << " this->getDimensions().y() + 1 = " << this->getDimensions().y() + 1 );
   tnlAssert( vertexCoordinates.z() >= 0 && vertexCoordinates.z() < this->getDimensions().z() + 1,
              cerr << "vertexCoordinates.z() = " << vertexCoordinates.z()
                   << " this->getDimensions().z() + 1 = " << this->getDimensions().z() + 1 );

   return ( vertexCoordinates.z() * ( this->getDimensions().y() + 1 ) + vertexCoordinates.y() ) *  ( this->getDimensions().x() + 1 ) + vertexCoordinates.x();
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
typename tnlGrid< 3, Real, Device, Index > :: CoordinatesType
tnlGrid< 3, Real, Device, Index > :: getVertexCoordinates( const Index vertexIndex ) const
{
   tnlAssert( vertexIndex >= 0 && vertexIndex < this->getNumberOfVertices(),
              cerr << " vertexIndex = " << vertexIndex
                   << " this->getNumberOfVertices() = " << this->getNumberOfVertices() );
   const IndexType auxX = this->getDimensions().x() + 1;
   const IndexType auxY = this->getDimensions().y() + 1;
   const IndexType auxZ = this->getDimensions().z() + 1;
   return CoordinatesType( vertexIndex % auxX,
                           ( vertexIndex / auxX ) % auxY,
                           vertexIndex / ( auxX * auxY ) );
}

template< typename Real,
          typename Device,
          typename Index >
   template< int dx, int dy, int dz >
__cuda_callable__
Index tnlGrid< 3, Real, Device, Index > :: getCellNextToCell( const IndexType& cellIndex ) const
{
   tnlAssert( cellIndex + dx >= 0 &&
              cellIndex + dx < this->getNumberOfCells() &&
              cellIndex + dy * this->getDimensions().x() >= 0 &&
              cellIndex + dy * this->getDimensions().x() < this->getNumberOfCells() &&
              cellIndex + dz * this->cellZNeighboursStep >= 0 &&
              cellIndex + dz * this->cellZNeighboursStep < this->getNumberOfCells(),
              cerr << " cellIndex = " << cellIndex
                   << " dx = " << dx
                   << " dy = " << dy
                   << " dz = " << dz
                   << " this->getNumberOfCells() = " << this->getNumberOfCells() );
   return cellIndex + dx +
          dy * this->getDimensions().x() +
          dz * this->cellZNeighboursStep;
}

template< typename Real,
          typename Device,
          typename Index >
   template< int nx, int ny, int nz >
__cuda_callable__
Index tnlGrid< 3, Real, Device, Index >::getFaceNextToCell( const IndexType& cellIndex ) const
{
   tnlAssert( nx * ny * nz == 0 && nx + ny + nz != 0,
              cerr << "nx = " << nx
                   << "ny = " << ny
                   << "nz = " << nz );
   IndexType result;
   if( nx )
      result = cellIndex + cellIndex / this->getDimensions().x() + ( nx + ( nx < 0 ) );
   if( ny )
      result = this->numberOfNxFaces + cellIndex + ( cellIndex / ( this->getDimensions().x() * this->getDimensions().y() ) + ( ny + ( ny < 0 ) ) )  * this->getDimensions().x();
   if( nz )
      result = this->numberOfNxAndNyFaces + cellIndex + ( nz + ( nz < 0 ) ) * this->getDimensions().x() * this->getDimensions().y();
   tnlAssert( result >= 0 &&
              result < this->getNumberOfFaces(),
              cerr << " cellIndex = " << cellIndex
                   << " nx = " << nx
                   << " ny = " << ny
                   << " nz = " << nz
                   << " this->getNumberOfCells() = " << this->getNumberOfCells() );
   return result;
}

template< typename Real,
          typename Device,
          typename Index >
   template< int nx, int ny, int nz >
__cuda_callable__
Index tnlGrid< 3, Real, Device, Index >::getCellNextToFace( const IndexType& faceIndex ) const
{
   tnlAssert( abs( nx ) + abs( ny ) + abs( nz ) == 1,
              cerr << "nx = " << nx << " ny = " << ny << " nz = " << nz );
#ifndef NDEBUG
   int _nx, _ny, _nz;
#endif
   tnlAssert( ( nx + this->getFaceCoordinates( faceIndex, _nx, _ny, _nz ).x() >= 0 &&
                nx + this->getFaceCoordinates( faceIndex, _nx, _ny, _nz ).x() <= this->getDimensions().x() ),
              cerr << " nx = " << nx
                   << " this->getFaceCoordinates( faceIndex, _nx, _ny, _nz ).x() = " << this->getFaceCoordinates( faceIndex, _nx, _ny, _nz ).x()
                   << " this->getDimensions().x()  = " << this->getDimensions().x() );
   tnlAssert( ( ny + this->getFaceCoordinates( faceIndex, _nx, _ny, _nz ).y() >= 0 &&
                ny + this->getFaceCoordinates( faceIndex, _nx, _ny, _nz ).y() <= this->getDimensions().y() ),
              cerr << " ny = " << ny
                   << " this->getFaceCoordinates( faceIndex, _nx, _ny, _nz ).y() = " << this->getFaceCoordinates( faceIndex, _nx, _ny, _nz ).y()
                   << " this->getDimensions().y()  = " << this->getDimensions().y() );
   tnlAssert( ( nz + this->getFaceCoordinates( faceIndex, _nx, _ny, _nz ).z() >= 0 &&
                nz + this->getFaceCoordinates( faceIndex, _nx, _ny, _nz ).z() <= this->getDimensions().z() ),
              cerr << " nz = " << nz
                   << " this->getFaceCoordinates( faceIndex, _nx, _ny, _nz ).z() = " << this->getFaceCoordinates( faceIndex, _nx, _ny, _nz ).z()
                   << " this->getDimensions().z()  = " << this->getDimensions().z() );

   IndexType result;
   if( nx )
      result = faceIndex + ( nx - ( nx > 0 ) ) - faceIndex / ( this->getDimensions().x() + 1 );
   if( ny )
   {
      IndexType aux = faceIndex - this->numberOfNxFaces;
      result = aux + ( ny - ( ny > 0 ) ) * this->getDimensions().x() - aux / ( ( this->getDimensions().y() + 1 ) * this->getDimensions().x() ) * this->getDimensions().x();
   }
   if( nz )
      result = faceIndex - this->numberOfNxAndNyFaces + ( nz - ( nz > 0 ) ) * this->getDimensions().y() * this->getDimensions().x();
   tnlAssert( result >= 0 &&
              result < this->getNumberOfFaces(),
              cerr << " faceIndex = " << faceIndex
                   << " nx = " << nx
                   << " ny = " << ny
                   << " nz = " << nz
                   << " this->getNumberOfCells() = " << this->getNumberOfCells() );
   return result;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 3, Real, Device, Index > :: getHx() const
{
   return this->hx;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 3, Real, Device, Index > :: getHxSquare() const
{
   return this->hxSquare;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 3, Real, Device, Index > :: getHxInverse() const
{
   return this->hxInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 3, Real, Device, Index > :: getHxSquareInverse() const
{
   return this->hxSquareInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 3, Real, Device, Index > :: getHy() const
{
   return this->hy;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 3, Real, Device, Index > :: getHySquare() const
{
   return this->hySquare;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 3, Real, Device, Index > :: getHyInverse() const
{
   return this->hyInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 3, Real, Device, Index > :: getHySquareInverse() const
{
   return this->hySquareInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 3, Real, Device, Index > :: getHz() const
{
   return this->hz;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 3, Real, Device, Index > :: getHzSquare() const
{
   return this->hzSquare;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 3, Real, Device, Index > :: getHzInverse() const
{
   return this->hzInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 3, Real, Device, Index > :: getHzSquareInverse() const
{
   return this->hzSquareInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 3, Real, Device, Index > :: getHxHy() const
{
   return this->hxhy;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 3, Real, Device, Index > :: getHxHz() const
{
   return this->hxhz;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 3, Real, Device, Index > :: getHyHz() const
{
   return this->hyhz;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 3, Real, Device, Index > :: getHxHyInverse() const
{
   return this->hxhyInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 3, Real, Device, Index > :: getHxHzInverse() const
{
   return this->hxhzInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 3, Real, Device, Index > :: getHyHzInverse() const
{
   return this->hyhzInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Real tnlGrid< 3, Real, Device, Index > :: getSmallestSpaceStep() const
{
   return Min( this->hx, Min( this->hy, this->hz ) );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityTopology,
             typename Vertex >
__cuda_callable__
Vertex tnlGrid< 3, Real, Device, Index > :: getEntityCenter( const CoordinatesType& coordinates ) const
{
   static_assert( EntityTopology::entityDimensions <= 3 &&
                  EntityTopology::entityDimensions >= 0, "Wrong grid entity dimensions." );
   tnlAssert( coordinates.x() >= 0 &&
              coordinates.x() <= this->getDimensions().x() - EntityTopology::EnityProportions::i1,
                    cerr << "coordinates.x() = " << coordinates.x()
                         << " this->getDimensions().x() = " << this->getDimensions().x()
                         << " EntityTopology::EntityProportions::i1 = " << EntityTopology::EntityProportions::i1 );
   tnlAssert( coordinates.y() >= 0 &&
              coordinates.y() < this->getDimensions().y() - EntityTopology::EnityProportions::i2,
                    cerr << "coordinates.y() = " << coordinates.y()
                         << " this->getDimensions().y() = " << this->getDimensions().y()
                         << " EntityTopology::EntityProportions::i2 = " << EntityTopology::EntityProportions::i2 );
   tnlAssert( coordinates.z() >= 0 &&
              coordinates.z() < this->getDimensions().z() - EntityTopology::EnityProportions::i3,
                    cerr << "coordinates.z() = " << coordinates.z()
                         << " this->getDimensions().z() = " << this->getDimensions().z()
                         << " EntityTopology::EntityProportions::i3 = " << EntityTopology::EntityProportions::i3 );

   return tnlGridEntityCenterGetter< ThisType, EntityTopology >::
      getCenter( coordinates, 
                 this->getOrigin(),
                 this->getCellProportions() );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityTopology,
             typename Vertex >
__cuda_callable__
Vertex tnlGrid< 3, Real, Device, Index > :: getEntityCenter( const IndexType& index ) const
{
   static_assert( EntityTopology::entityDimensions <= 3 &&
                  EntityTopology::entityDimensions >= 0, "Wrong grid entity dimensions." );

   if( EntityTopology::entityDimensions == Dimensions )
      return this->getCellCenter( index );
   if( EntityTopology::entityDimensions == Dimensions - 1 )
      return this->template getFaceCenter< EntityTopology::EntityOrientation::i1,
                                           EntityTopology::EntityOrientation::i2,
                                           EntityTopology::EntityOrientation::i3 >( index );
   if( EntityTopology::entityDimensions == Dimensions - 2 )
      return this->template getEdgeCenter< EntityTopology::EntityOrientation::i1,
                                           EntityTopology::EntityOrientation::i2,
                                           EntityTopology::EntityOrientation::i3 >( index );
   if( EntityTopology::entityDimensions == Dimensions - 3 )
      return this->template getVertex( index );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vertex >
__cuda_callable__
Vertex tnlGrid< 3, Real, Device, Index > :: getCellCenter( const CoordinatesType& cellCoordinates ) const
{
   tnlAssert( cellCoordinates.x() >= 0 && cellCoordinates.x() < this->getDimensions().x(),
              cerr << "cellCoordinates.x() = " << cellCoordinates.x()
                   << " this->getDimensions().x() = " << this->getDimensions().x() );
   tnlAssert( cellCoordinates.y() >= 0 && cellCoordinates.y() < this->getDimensions().y(),
              cerr << "cellCoordinates.y() = " << cellCoordinates.y()
                   << " this->getDimensions().y() = " << this->getDimensions().y() );
   tnlAssert( cellCoordinates.z() >= 0 && cellCoordinates.z() < this->getDimensions().z(),
              cerr << "cellCoordinates.z() = " << cellCoordinates.z()
                   << " this->getDimensions().z() = " << this->getDimensions().z() );


   return Vertex( this->origin.x() + ( cellCoordinates.x() + 0.5 ) * this->cellProportions.x(),
                  this->origin.y() + ( cellCoordinates.y() + 0.5 ) * this->cellProportions.y(),
                  this->origin.z() + ( cellCoordinates.z() + 0.5 ) * this->cellProportions.z() );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vertex >
__cuda_callable__
Vertex tnlGrid< 3, Real, Device, Index >::getCellCenter( const IndexType& cellIndex ) const
{
   tnlAssert( cellIndex >= 0 && cellIndex < this->getNumberOfCells(),
              cerr << " cellIndex = " << cellIndex
                   << " this->getNumberOfCells() = " << this->getNumberOfCells() );
   return this->getCellCenter< VertexType >( this->getCellCoordinates( cellIndex ) );
}

template< typename Real,
          typename Device,
          typename Index >
template< int nx, int ny, int nz, typename Vertex >
__cuda_callable__
Vertex tnlGrid< 3, Real, Device, Index > :: getFaceCenter( const CoordinatesType& faceCoordinates ) const
{
   tnlStaticAssert( nx >= 0 && ny >= 0 && nz >= 0 && nx + ny + nz == 1, "Wrong template parameters nx or ny or nz." );
   if( nx )
   {
      tnlAssert( faceCoordinates.x() >= 0 && faceCoordinates.x() < this->getDimensions().x() + 1,
                 cerr << "faceCoordinates.x() = " << faceCoordinates.x()
                      << " this->getDimensions().x() + 1 = " << this->getDimensions().x() + 1 );
      tnlAssert( faceCoordinates.y() >= 0 && faceCoordinates.y() < this->getDimensions().y(),
                 cerr << "faceCoordinates.y() = " << faceCoordinates.y()
                      << " this->getDimensions().y() = " << this->getDimensions().y() );
      tnlAssert( faceCoordinates.z() >= 0 && faceCoordinates.z() < this->getDimensions().z(),
                 cerr << "faceCoordinates.z() = " << faceCoordinates.z()
                      << " this->getDimensions().z() = " << this->getDimensions().z() );
      return Vertex( this->origin.x() + faceCoordinates.x() * this->cellProportions().x(),
                     this->origin.y() + ( faceCoordinates.y() + 0.5 ) * this->cellProportions().y(),
                     this->origin.z() + ( faceCoordinates.y() + 0.5 ) * this->cellProportions().z() );
   }
   if( ny )
   {
      tnlAssert( faceCoordinates.x() >= 0 && faceCoordinates.x() < this->getDimensions().x(),
                 cerr << "faceCoordinates.x() = " << faceCoordinates.x()
                      << " this->getDimensions().x() = " << this->getDimensions().x() );
      tnlAssert( faceCoordinates.y() >= 0 && faceCoordinates.y() < this->getDimensions().y() + 1,
                 cerr << "faceCoordinates.y() = " << faceCoordinates.y()
                      << " this->getDimensions().y() + 1 = " << this->getDimensions().y() + 1 );
      tnlAssert( faceCoordinates.z() >= 0 && faceCoordinates.z() < this->getDimensions().z(),
                 cerr << "faceCoordinates.z() = " << faceCoordinates.z()
                      << " this->getDimensions().z() = " << this->getDimensions().z() );

      return Vertex( this->origin.x() + ( faceCoordinates.x() + 0.5 ) * this->cellProportions().x(),
                     this->origin.y() + faceCoordinates.y() * this->cellProportions().y(),
                     this->origin.z() + ( faceCoordinates.z() + 0.5 ) * this->cellProportions().z() );
   }
   if( nz )
   {
      tnlAssert( faceCoordinates.x() >= 0 && faceCoordinates.x() < this->getDimensions().x(),
                 cerr << "faceCoordinates.x() = " << faceCoordinates.x()
                      << " this->getDimensions().x() = " << this->getDimensions().x() );
      tnlAssert( faceCoordinates.y() >= 0 && faceCoordinates.y() < this->getDimensions().y(),
                 cerr << "faceCoordinates.y() = " << faceCoordinates.y()
                      << " this->getDimensions().y()= " << this->getDimensions().y() );
      tnlAssert( faceCoordinates.z() >= 0 && faceCoordinates.z() < this->getDimensions().z() + 1,
                 cerr << "faceCoordinates.z() = " << faceCoordinates.z()
                      << " this->getDimensions().z() + 1 = " << this->getDimensions().z() + 1 );
      return Vertex( this->origin.x() + ( faceCoordinates.x() + 0.5 ) * this->cellProportions().x(),
                     this->origin.y() + ( faceCoordinates.y() + 0.5 ) * this->cellProportions().y(),
                     this->origin.z() + faceCoordinates.z() * this->cellProportions().z() );
   }
}

template< typename Real,
          typename Device,
          typename Index >
template< int dx, int dy, int dz, typename Vertex >
__cuda_callable__
Vertex tnlGrid< 3, Real, Device, Index > :: getEdgeCenter( const CoordinatesType& edgeCoordinates ) const
{
   tnlStaticAssert( dx >= 0 && dy >= 0 && dz >= 0 && dx + dy + dz == 1, "Wrong template parameters nx or ny or nz." );
   if( dx )
   {
      tnlAssert( edgeCoordinates.x() >= 0 && edgeCoordinates.x() < this->getDimensions().x(),
                 cerr << "edgeCoordinates.x() = " << edgeCoordinates.x()
                      << " this->getDimensions().x() = " << this->getDimensions().x() );
      tnlAssert( edgeCoordinates.y() >= 0 && edgeCoordinates.y() < this->getDimensions().y() + 1,
                 cerr << "edgeCoordinates.y() = " << edgeCoordinates.y()
                      << " this->getDimensions().y() + 1 = " << this->getDimensions().y() + 1 );
      tnlAssert( edgeCoordinates.z() >= 0 && edgeCoordinates.z() < this->getDimensions().z() + 1,
                 cerr << "edgeCoordinates.z() = " << edgeCoordinates.z()
                      << " this->getDimensions().z() + 1 = " << this->getDimensions().z() + 1 );
      return Vertex( this->origin.x() + ( edgeCoordinates.x() + 0.5 ) * this->cellProportions().x(),
                     this->origin.y() + edgeCoordinates.y() * this->cellProportions().y(),
                     this->origin.z() + edgeCoordinates.z() * this->cellProportions().z() );
   }
   if( dy )
   {
      tnlAssert( edgeCoordinates.x() >= 0 && edgeCoordinates.x() < this->getDimensions().x() + 1,
                 cerr << "edgeCoordinates.x() = " << edgeCoordinates.x()
                      << " this->getDimensions().x() + 1 = " << this->getDimensions().x() + 1 );
      tnlAssert( edgeCoordinates.y() >= 0 && edgeCoordinates.y() < this->getDimensions().y(),
                 cerr << "edgeCoordinates.y() = " << edgeCoordinates.y()
                      << " this->getDimensions().y() = " << this->getDimensions().y() );
      tnlAssert( edgeCoordinates.z() >= 0 && edgeCoordinates.z() < this->getDimensions().z() + 1,
                 cerr << "edgeCoordinates.z() = " << edgeCoordinates.z()
                      << " this->getDimensions().z() + 1 = " << this->getDimensions().z() + 1 );
      return Vertex( this->origin.x() + edgeCoordinates.x() * this->cellProportions().x(),
                     this->origin.y() + ( edgeCoordinates.y() + 0.5 ) * this->cellProportions().y(),
                     this->origin.z() + edgeCoordinates.z() * this->cellProportions().z() );
   }
   if( dz )
   {
      tnlAssert( edgeCoordinates.x() >= 0 && edgeCoordinates.x() < this->getDimensions().x() + 1,
                 cerr << "edgeCoordinates.x() = " << edgeCoordinates.x()
                      << " this->getDimensions().x() = " << this->getDimensions().x() );
      tnlAssert( edgeCoordinates.y() >= 0 && edgeCoordinates.y() < this->getDimensions().y() + 1,
                 cerr << "edgeCoordinates.y() = " << edgeCoordinates.y()
                      << " this->getDimensions().y() + 1 = " << this->getDimensions().y() + 1 );
      tnlAssert( edgeCoordinates.z() >= 0 && edgeCoordinates.z() < this->getDimensions().z(),
                 cerr << "edgeCoordinates.z() = " << edgeCoordinates.z()
                      << " this->getDimensions().z() = " << this->getDimensions().z() );
      return Vertex( this->origin.x() + edgeCoordinates.x() * this->cellProportions().x(),
                     this->origin.y() + edgeCoordinates.y() * this->cellProportions().y(),
                     this->origin.z() + ( edgeCoordinates.z() + 0.5 ) * this->cellProportions().z() );
   }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vertex >
__cuda_callable__
Vertex tnlGrid< 3, Real, Device, Index >::getVertex( const CoordinatesType& vertexCoordinates ) const
{
   tnlAssert( vertexCoordinates.x() >= 0 && vertexCoordinates.x() < this->getDimensions().x() + 1,
              cerr << "vertexCoordinates.x() = " << vertexCoordinates.x()
                   << " this->getDimensions().x() = " << this->getDimensions().x() );
   tnlAssert( vertexCoordinates.y() >= 0 && vertexCoordinates.y() < this->getDimensions().y() + 1,
              cerr << "vertexCoordinates.y() = " << vertexCoordinates.y()
                   << " this->getDimensions().y() = " << this->getDimensions().y() );
   tnlAssert( vertexCoordinates.z() >= 0 && vertexCoordinates.z() < this->getDimensions().z() + 1,
              cerr << "vertexCoordinates.z() = " << vertexCoordinates.z()
                   << " this->getDimensions().z() = " << this->getDimensions().z() );

   return Vertex( this->origin.x() + vertexCoordinates.x() * this->cellProportions.x(),
                  this->origin.y() + vertexCoordinates.y() * this->cellProportions.y(),
                  this->origin.z() + vertexCoordinates.z() * this->cellProportions.z() );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index tnlGrid< 3, Real, Device, Index > :: getNumberOfCells() const
{
   return this->numberOfCells;
};

template< typename Real,
          typename Device,
          typename Index >
   template< int nx, int ny, int nz >
__cuda_callable__
Index tnlGrid< 3, Real, Device, Index > :: getNumberOfFaces() const
{
   return nx * this->numberOfNxFaces +
          ny * this->numberOfNyFaces +
          nz * this->numberOfNzFaces;
}

template< typename Real,
          typename Device,
          typename Index >
   template< int dx, int dy, int dz >
__cuda_callable__
Index tnlGrid< 3, Real, Device, Index > :: getNumberOfEdges() const
{
   return dx * this->numberOfDxEdges +
          dy * this->numberOfDyEdges +
          dz * this->numberOfDzEdges;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index tnlGrid< 3, Real, Device, Index > :: getNumberOfVertices() const
{
   return numberOfVertices;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool tnlGrid< 3, Real, Device, Index > :: isBoundaryCell( const CoordinatesType& cellCoordinates ) const
{
   tnlAssert( cellCoordinates.x() >= 0 && cellCoordinates.x() < this->getDimensions().x(),
              cerr << "cellCoordinates.x() = " << cellCoordinates.x()
                   << " this->getDimensions().x() = " << this->getDimensions().x() );
   tnlAssert( cellCoordinates.y() >= 0 && cellCoordinates.y() < this->getDimensions().y(),
              cerr << "cellCoordinates.y() = " << cellCoordinates.y()
                   << " this->getDimensions().y() = " << this->getDimensions().y() );
   tnlAssert( cellCoordinates.z() >= 0 && cellCoordinates.z() < this->getDimensions().z(),
              cerr << "cellCoordinates.z() = " << cellCoordinates.z()
                   << " this->getDimensions().z() = " << this->getDimensions().z() );


   if( cellCoordinates.x() == 0 || cellCoordinates.x() == this->getDimensions().x() - 1 ||
       cellCoordinates.y() == 0 || cellCoordinates.y() == this->getDimensions().y() - 1 ||
       cellCoordinates.z() == 0 || cellCoordinates.z() == this->getDimensions().z() - 1 )
      return true;
   return false;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool
tnlGrid< 3, Real, Device, Index >::
isBoundaryCell( const IndexType& cellIndex ) const
{
   tnlAssert( cellIndex >= 0 && cellIndex < this->getNumberOfCells(),
              cerr << " cellIndex = " << cellIndex
                   << " this->getNumberOfCells() = " << this->getNumberOfCells() );
   return this->isBoundaryCell( this->getCellCoordinates( cellIndex ) );
}

template< typename Real,
          typename Device,
          typename Index >
   template< int nx, int ny, int nz >
__cuda_callable__
bool tnlGrid< 3, Real, Device, Index > :: isBoundaryFace( const CoordinatesType& faceCoordinates ) const
{
   tnlStaticAssert( nx >= 0 && ny >= 0 && nz >=0 && nx + ny + nz == 1, "Wrong template parameters nx or ny or nz." );
   if( nx )
   {
      tnlAssert( faceCoordinates.x() >= 0 && faceCoordinates.x() < this->getDimensions().x() + 1,
                 cerr << "faceCoordinates.x() = " << faceCoordinates.x()
                      << " this->getDimensions().x() + 1 = " << this->getDimensions().x() + 1 );
      tnlAssert( faceCoordinates.y() >= 0 && faceCoordinates.y() < this->getDimensions().y(),
                 cerr << "faceCoordinates.y() = " << faceCoordinates.y()
                      << " this->getDimensions().y() = " << this->getDimensions().y() );
      if( faceCoordinates.x() == 0 || faceCoordinates.x() == this->getDimensions().y() )
         return true;
      return false;
   }
   if( ny )
   {
      tnlAssert( faceCoordinates.x() >= 0 && faceCoordinates.x() < this->getDimensions().x(),
                 cerr << "faceCoordinates.x() = " << faceCoordinates.x()
                      << " this->getDimensions().x() = " << this->getDimensions().x() );
      tnlAssert( faceCoordinates.y() >= 0 && faceCoordinates.y() < this->getDimensions().y() + 1,
                 cerr << "faceCoordinates.y() = " << faceCoordinates.y()
                      << " this->getDimensions().y() + 1 = " << this->getDimensions().y() + 1 );
      if( faceCoordinates.y() == 0 || faceCoordinates.y() == this->getDimensions().y() )
         return true;
      return false;
   }
   if( nz )
   {
      tnlAssert( faceCoordinates.x() >= 0 && faceCoordinates.x() < this->getDimensions().x(),
                 cerr << "faceCoordinates.x() = " << faceCoordinates.x()
                      << " this->getDimensions().x() = " << this->getDimensions().x() );
      tnlAssert( faceCoordinates.y() >= 0 && faceCoordinates.y() < this->getDimensions().y(),
                 cerr << "faceCoordinates.y() = " << faceCoordinates.y()
                      << " this->getDimensions().y()= " << this->getDimensions().y() );
      tnlAssert( faceCoordinates.z() >= 0 && faceCoordinates.z() < this->getDimensions().z() + 1,
                 cerr << "faceCoordinates.z() = " << faceCoordinates.z()
                      << " this->getDimensions().z() + 1 = " << this->getDimensions().z() + 1 );
      if( faceCoordinates.z() == 0 || faceCoordinates.z() == this->getDimensions().z() )
         return true;
      return false;
   }   
}

template< typename Real,
          typename Device,
          typename Index >
   template< int dx, int dy, int dz >
__cuda_callable__
bool tnlGrid< 3, Real, Device, Index >::isBoundaryEdge( const CoordinatesType& edgeCoordinates ) const
{
   tnlStaticAssert( dx >= 0 && dy >= 0 && dz >= 0 && dx + dy + dz == 1, "Wrong template parameters nx or ny or nz." );
   if( dx )
   {
      tnlAssert( edgeCoordinates.x() >= 0 && edgeCoordinates.x() < this->getDimensions().x(),
                 cerr << "edgeCoordinates.x() = " << edgeCoordinates.x()
                      << " this->getDimensions().x() = " << this->getDimensions().x() );
      tnlAssert( edgeCoordinates.y() >= 0 && edgeCoordinates.y() < this->getDimensions().y() + 1,
                 cerr << "edgeCoordinates.y() = " << edgeCoordinates.y()
                      << " this->getDimensions().y() + 1 = " << this->getDimensions().y() + 1 );
      tnlAssert( edgeCoordinates.z() >= 0 && edgeCoordinates.z() < this->getDimensions().z() + 1,
                 cerr << "edgeCoordinates.z() = " << edgeCoordinates.z()
                      << " this->getDimensions().z() + 1 = " << this->getDimensions().z() + 1 );
      if( edgeCoordinates.y() == 0 || edgeCoordinates.y() == this->getDimensions().y() ||
          edgeCoordinates.z() == 0 || edgeCoordinates.z() == this->getDimensions().z() )
         return true;
      return false;
   }
   if( dy )
   {
      tnlAssert( edgeCoordinates.x() >= 0 && edgeCoordinates.x() < this->getDimensions().x() + 1,
                 cerr << "edgeCoordinates.x() = " << edgeCoordinates.x()
                      << " this->getDimensions().x() + 1 = " << this->getDimensions().x() + 1 );
      tnlAssert( edgeCoordinates.y() >= 0 && edgeCoordinates.y() < this->getDimensions().y(),
                 cerr << "edgeCoordinates.y() = " << edgeCoordinates.y()
                      << " this->getDimensions().y() = " << this->getDimensions().y() );
      tnlAssert( edgeCoordinates.z() >= 0 && edgeCoordinates.z() < this->getDimensions().z() + 1,
                 cerr << "edgeCoordinates.z() = " << edgeCoordinates.z()
                      << " this->getDimensions().z() + 1 = " << this->getDimensions().z() + 1 );
      if( edgeCoordinates.x() == 0 || edgeCoordinates.x() == this->getDimensions().x() ||
          edgeCoordinates.z() == 0 || edgeCoordinates.z() == this->getDimensions().z() )
         return true;
      return false;

   }
   if( dz )
   {
      tnlAssert( edgeCoordinates.x() >= 0 && edgeCoordinates.x() < this->getDimensions().x() + 1,
                 cerr << "edgeCoordinates.x() = " << edgeCoordinates.x()
                      << " this->getDimensions().x() = " << this->getDimensions().x() );
      tnlAssert( edgeCoordinates.y() >= 0 && edgeCoordinates.y() < this->getDimensions().y() + 1,
                 cerr << "edgeCoordinates.y() = " << edgeCoordinates.y()
                      << " this->getDimensions().y() + 1 = " << this->getDimensions().y() + 1 );
      tnlAssert( edgeCoordinates.z() >= 0 && edgeCoordinates.z() < this->getDimensions().z(),
                 cerr << "edgeCoordinates.z() = " << edgeCoordinates.z()
                      << " this->getDimensions().z() = " << this->getDimensions().z() );
      if( edgeCoordinates.x() == 0 || edgeCoordinates.x() == this->getDimensions().x() ||
          edgeCoordinates.y() == 0 || edgeCoordinates.y() == this->getDimensions().y() )
         return true;
      return false;
   }
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool tnlGrid< 3, Real, Device, Index > :: isBoundaryVertex( const CoordinatesType& vertexCoordinates ) const
{
   tnlAssert( vertexCoordinates.x() >= 0 && vertexCoordinates.x() < this->getDimensions().x() + 1,
              cerr << "vertexCoordinates.x() = " << vertexCoordinates.x()
                   << " this->getDimensions().x() + 1 = " << this->getDimensions().x() + 1 );
   tnlAssert( vertexCoordinates.y() >= 0 && vertexCoordinates.y() < this->getDimensions().y() + 1,
              cerr << "vertexCoordinates.y() = " << vertexCoordinates.y()
                   << " this->getDimensions().y() + 1 = " << this->getDimensions().y() + 1 );
   tnlAssert( vertexCoordinates.z() >= 0 && vertexCoordinates.z() < this->getDimensions().z() + 1,
              cerr << "vertexCoordinates.z() = " << vertexCoordinates.z()
                   << " this->getDimensions().z() + 1 = " << this->getDimensions().z() + 1 );

   if( vertexCoordinates.x() == 0 || vertexCoordinates.x() == this->getDimensions().x() ||
       vertexCoordinates.y() == 0 || vertexCoordinates.y() == this->getDimensions().y() ||
       vertexCoordinates.z() == 0 || vertexCoordinates.z() == this->getDimensions().z() )
      return true;
   return false;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
typename GridFunction::RealType
   tnlGrid< 3, Real, Device, Index >::getAbsMax( const GridFunction& f ) const
{
   return f.absMax();
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
typename GridFunction::RealType
   tnlGrid< 3, Real, Device, Index >::getLpNorm( const GridFunction& f1,
                                                 const typename GridFunction::RealType& p ) const
{
   typename GridFunction::RealType lpNorm( 0.0 );
   for( IndexType k = 0; k < getDimensions().z(); k++ )
      for( IndexType j = 0; j < getDimensions().y(); j++ )
         for( IndexType i = 0; i < getDimensions().x(); i++ )
         {
            IndexType c = this->getElementIndex( i, j, k );
            lpNorm += pow( tnlAbs( f1[ c ] ), p );;
         }
   lpNorm *= this->cellProportions().x() * this->cellProportions().y() * this->cellProportions().z();
   return pow( lpNorm, 1.0/p );
}


template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
      typename GridFunction::RealType
         tnlGrid< 3, Real, Device, Index >::getDifferenceAbsMax( const GridFunction& f1,
                                                                           const GridFunction& f2 ) const
{
   typename GridFunction::RealType maxDiff( -1.0 );
   for( IndexType k = 0; k < getDimensions(). z(); k++ )
      for( IndexType j = 0; j < getDimensions(). y(); j++ )
         for( IndexType i = 0; i < getDimensions(). x(); i++ )
         {
            IndexType c = this -> getCellIndex( CoordinatesType( i, j, k ) );
            maxDiff = Max( maxDiff, tnlAbs( f1[ c ] - f2[ c ] ) );
         }
   return maxDiff;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
      typename GridFunction::RealType
         tnlGrid< 3, Real, Device, Index >::getDifferenceLpNorm( const GridFunction& f1,
                                                                           const GridFunction& f2,
                                                                           const typename GridFunction::RealType& p ) const
{
   typename GridFunction::RealType lpNorm( 0.0 );
   for( IndexType k = 0; k < getDimensions(). z(); k++ )
      for( IndexType j = 0; j < getDimensions(). y(); j++ )
         for( IndexType i = 0; i < getDimensions(). x(); i++ )
         {
            IndexType c = this->getCellIndex( CoordinatesType( i, j, k ) );
            lpNorm += pow( tnlAbs( f1[ c ] - f2[ c ] ), p );
         }
   lpNorm *= this->cellProportions.x() * this->cellProportions.y() * this->cellProportions.z();
   return pow( lpNorm, 1.0 / p );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlGrid< 3, Real, Device, Index > :: save( tnlFile& file ) const
{
   if( ! tnlObject::save( file ) )
      return false;
   if( ! this->origin.save( file ) ||
       ! this->proportions.save( file ) ||
       ! this->dimensions.save( file ) )
   {
      cerr << "I was not able to save the domain description of a tnlGrid." << endl;
      return false;
   }
   return true;
};

template< typename Real,
          typename Device,
          typename Index >
bool tnlGrid< 3, Real, Device, Index > :: load( tnlFile& file )
{
   if( ! tnlObject :: load( file ) )
      return false;
   CoordinatesType dimensions;
   if( ! this->origin.load( file ) ||
       ! this->proportions.load( file ) ||
       ! dimensions.load( file ) )
   {
      cerr << "I was not able to load the domain description of a tnlGrid." << endl;
      return false;
   }
   this->setDimensions( dimensions );
   return true;
};

template< typename Real,
          typename Device,
          typename Index >
bool tnlGrid< 3, Real, Device, Index > :: save( const tnlString& fileName ) const
{
   return tnlObject :: save( fileName );
};

template< typename Real,
          typename Device,
          typename Index >
bool tnlGrid< 3, Real, Device, Index > :: load( const tnlString& fileName )
{
   return tnlObject :: load( fileName );
};

template< typename Real,
           typename Device,
           typename Index >
bool tnlGrid< 3, Real, Device, Index >::writeMesh( const tnlString& fileName,
                                                   const tnlString& format ) const
{
   tnlAssert( false, cerr << "TODO: FIX THIS"); // TODO: FIX THIS
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename MeshFunction >
bool tnlGrid< 3, Real, Device, Index > :: write( const MeshFunction& function,
                                                 const tnlString& fileName,
                                                 const tnlString& format ) const
{
   if( this -> getNumberOfCells() != function. getSize() )
   {
      cerr << "The size ( " << function. getSize() 
           << " ) of a mesh function does not agree with the DOFs ( " << this -> getNumberOfCells() << " ) of a mesh." << endl;
      return false;
   }
   fstream file;
   file. open( fileName. getString(), ios :: out );
   if( ! file )
   {
      cerr << "I am not able to open the file " << fileName << "." << endl;
      return false;
   }
   file << setprecision( 12 );
   if( format == "gnuplot" )
   {
      for( IndexType k = 0; k < getDimensions(). z(); k++ )
      {
         for( IndexType j = 0; j < getDimensions(). y(); j++ )
         {
            for( IndexType i = 0; i < getDimensions(). x(); i++ )
            {
               VertexType v = this -> getCellCenter< VertexType >( CoordinatesType( i, j, k ) );
               tnlGnuplotWriter::write( file, v );
               tnlGnuplotWriter::write( file, function[ this->getCellIndex( CoordinatesType( i, j, k ) ) ] );
               file << endl;
            }
         }
         file << endl;
      }
   }

   file. close();
   return true;
}

template< typename Real,
           typename Device,
           typename Index >
void
tnlGrid< 3, Real, Device, Index >::
writeProlog( tnlLogger& logger )
{
   logger.writeParameter( "Dimensions:", Dimensions );
   logger.writeParameter( "Domain origin:", this->origin );
   logger.writeParameter( "Domain proportions:", this->proportions );
   logger.writeParameter( "Domain dimensions:", this->dimensions );
   logger.writeParameter( "Cell proportions:", this->cellProportions );
}


#endif /* TNLGRID3D_IMPL_H_ */
