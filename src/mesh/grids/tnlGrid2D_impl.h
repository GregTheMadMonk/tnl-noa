/***************************************************************************
                          tnlGrid2D_impl.h  -  description
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

#ifndef TNLGRID2D_IMPL_H_
#define TNLGRID2D_IMPL_H_

#include <fstream>
#include <iomanip>
#include <core/tnlAssert.h>
#include <mesh/tnlGnuplotWriter.h>
#include <mesh/grids/tnlGridEntityGetter_impl.h>
#include <mesh/grids/tnlNeighbourGridEntityGetter2D_impl.h>

using namespace std;

template< typename Real,
          typename Device,
          typename Index >
tnlGrid< 2, Real, Device, Index > :: tnlGrid()
: numberOfCells( 0 ),
  numberOfNxFaces( 0 ),
  numberOfNyFaces( 0 ),
  numberOfFaces( 0 ),
  numberOfVertices( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlGrid< 2, Real, Device, Index > :: getType()
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
tnlString tnlGrid< 2, Real, Device, Index > :: getTypeVirtual() const
{
   return this -> getType();
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlGrid< 2, Real, Device, Index > :: getSerializationType()
{
   return HostType::getType();
};

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlGrid< 2, Real, Device, Index > :: getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
void tnlGrid< 2, Real, Device, Index > :: computeSpaceSteps()
{
   if( this->getDimensions().x() > 0 && this->getDimensions().y() > 0 )
   {
      this->cellProportions.x() = this->proportions.x() / ( Real ) this->getDimensions().x();
      this->cellProportions.y() = this->proportions.y() / ( Real ) this->getDimensions().y();
      this->hx = this->proportions.x() / ( Real ) this->getDimensions().x();
      this->hxSquare = this->hx * this->hx;
      this->hxInverse = 1.0 / this->hx;
      this->hxSquareInverse = this->hxInverse * this->hxInverse;
      this->hy = this->proportions.y() / ( Real ) this->getDimensions().y();
      this->hySquare = this->hy * this->hy;
      this->hyInverse = 1.0 / this->hy;
      this->hySquareInverse = this->hyInverse * this->hyInverse;
      this->hxhy = this->hx * this->hy;
      this->hxhyInverse = 1.0 / this->hxhy;
   }
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 2, Real, Device, Index > :: setDimensions( const Index xSize, const Index ySize )
{
   tnlAssert( xSize > 0, cerr << "xSize = " << xSize );
   tnlAssert( ySize > 0, cerr << "ySize = " << ySize );

   this->dimensions.x() = xSize;
   this->dimensions.y() = ySize;
   this->numberOfCells = xSize * ySize;
   this->numberOfNxFaces = ySize * ( xSize + 1 );
   this->numberOfNyFaces = xSize * ( ySize + 1 );
   this->numberOfFaces = this->numberOfNxFaces + this->numberOfNyFaces;
   this->numberOfVertices = ( xSize + 1 ) * ( ySize + 1 );
   computeSpaceSteps();
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 2, Real, Device, Index > :: setDimensions( const CoordinatesType& dimensions )
{
   return this->setDimensions( dimensions. x(), dimensions. y() );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const typename tnlGrid< 2, Real, Device, Index >::CoordinatesType&
tnlGrid< 2, Real, Device, Index > :: getDimensions() const
{
   return this->dimensions;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 2, Real, Device, Index > :: setDomain( const VertexType& origin,
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
const typename tnlGrid< 2, Real, Device, Index >::VertexType&
tnlGrid< 2, Real, Device, Index >::getOrigin() const
{
   return this->origin;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const typename tnlGrid< 2, Real, Device, Index > :: VertexType&
   tnlGrid< 2, Real, Device, Index > :: getProportions() const
{
   return this->proportions;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const typename tnlGrid< 2, Real, Device, Index > :: VertexType&
tnlGrid< 2, Real, Device, Index > :: getCellProportions() const
{
   return this->cellProportions;
}

template< typename Real,
          typename Device,
          typename Index >
   template< int EntityDimensions >
Index
tnlGrid< 2, Real, Device, Index >:: 
getEntitiesCount() const
{
   static_assert( EntityDimensions <= 2 &&
                  EntityDimensions >= 0, "Wrong grid entity dimensions." );
   
   switch( EntityDimensions )
   {
      case 2:
         return this->numberOfCells;
      case 1:
         return this->numberOfFaces;         
      case 0:
         return this->numberOfVertices;
   }            
}

template< typename Real,
          typename Device,
          typename Index >
   template< int EntityDimensions >
 __cuda_callable__
typename tnlGrid< 2, Real, Device, Index >::template GridEntity< EntityDimensions >
tnlGrid< 2, Real, Device, Index >::
getEntity( const IndexType& entityIndex ) const
{
   static_assert( EntityDimensions <= 2 &&
                  EntityDimensions >= 0, "Wrong grid entity dimensions." );
   
   return tnlGridEntityGetter< ThisType, EntityDimensions >::getEntity( *this, entityIndex );
}

template< typename Real,
          typename Device,
          typename Index >
   template< int EntityDimensions >
__cuda_callable__
Index
tnlGrid< 2, Real, Device, Index >::
getEntityIndex( const GridEntity< EntityDimensions >& entity ) const
{
   static_assert( EntityDimensions <= 2 &&
                  EntityDimensions >= 0, "Wrong grid entity dimensions." );
   
   return tnlGridEntityGetter< ThisType, EntityDimensions >::getEntityIndex( *this, entity );
}

template< typename Real,
          typename Device,
          typename Index >
   template< int EntityDimensions,
             typename Vertex >
__cuda_callable__
Vertex tnlGrid< 2, Real, Device, Index > :: getEntityCenter( const GridEntity< EntityDimensions >& entity ) const
{
   static_assert( EntityDimensions <= 2 &&
                  EntityDimensions >= 0, "Wrong grid entity dimensions." );
   tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0 ) &&
              entity.getCoordinates() <= this->getDimensions() - entity.getBasis(),
                    cerr << "entity.getCoordinates(). = " << entity.getCoordinates()
                         << " this->getDimensions() = " << this->getDimensions()
                         << " entity.getBasis() = " << entity.getBasis() );
   return Vertex( this->origin.x() + ( entity.getCoordinates().x() + 0.5 * entity.getBasis().x() ) * this->cellProportions.x(),
                  this->origin.y() + ( entity.getCoordinates().y() + 0.5 * entity.getBasis().y() ) * this->cellProportions.y() );
}






template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index tnlGrid< 2, Real, Device, Index > :: getCellIndex( const CoordinatesType& cellCoordinates ) const
{
   tnlAssert( cellCoordinates.x() >= 0 && cellCoordinates.x() < this->getDimensions().x(),
              cerr << "cellCoordinates.x() = " << cellCoordinates.x()
                   << " this->getDimensions().x() = " << this->getDimensions().x() );
   tnlAssert( cellCoordinates.y() >= 0 && cellCoordinates.y() < this->getDimensions().y(),
              cerr << "cellCoordinates.y() = " << cellCoordinates.y()
                   << " this->getDimensions().y() = " << this->getDimensions().y() )

   return cellCoordinates.y() * this->dimensions.x() + cellCoordinates.x();
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
typename tnlGrid< 2, Real, Device, Index >::CoordinatesType
tnlGrid< 2, Real, Device, Index >::getCellCoordinates( const Index cellIndex ) const
{
   tnlAssert( cellIndex >= 0 && cellIndex < this->template getEntitiesCount< Cells >(),
              cerr << " cellIndex = " << cellIndex
                   << " this->template getEntitiesCount< Cells >() = " << this->template getEntitiesCount< Cells >() );
   return CoordinatesType( cellIndex % this->getDimensions().x(), cellIndex / this->getDimensions().x() );
}

template< typename Real,
          typename Device,
          typename Index >
template< int nx, int ny >
__cuda_callable__
Index tnlGrid< 2, Real, Device, Index >::getFaceIndex( const CoordinatesType& faceCoordinates ) const
{
   tnlStaticAssert( nx >= 0 && ny >= 0 && nx + ny == 1, "Wrong template parameters nx or ny." );
   if( nx )
   {
      tnlAssert( faceCoordinates.x() >= 0 && faceCoordinates.x() < this->getDimensions().x() + 1,
                 cerr << "faceCoordinates.x() = " << faceCoordinates.x()
                      << " this->getDimensions().x() + 1 = " << this->getDimensions().x() + 1 );
      tnlAssert( faceCoordinates.y() >= 0 && faceCoordinates.y() < this->getDimensions().y(),
                 cerr << "faceCoordinates.y() = " << faceCoordinates.y()
                      << " this->getDimensions().y() = " << this->getDimensions().y() );
      return faceCoordinates.y() * ( this->getDimensions().x() + 1 ) + faceCoordinates.x();
   }
   tnlAssert( faceCoordinates.x() >= 0 && faceCoordinates.x() < this->getDimensions().x(),
              cerr << "faceCoordinates.x() = " << faceCoordinates.x()
                   << " this->getDimensions().x() = " << this->getDimensions().x() );
   tnlAssert( faceCoordinates.y() >= 0 && faceCoordinates.y() < this->getDimensions().y() + 1,
              cerr << "faceCoordinates.y() = " << faceCoordinates.y()
                   << " this->getDimensions().y() + 1 = " << this->getDimensions().y() + 1 );
   return this->numberOfNxFaces + faceCoordinates.y() * this->getDimensions().x() + faceCoordinates.x();
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
typename tnlGrid< 2, Real, Device, Index >::CoordinatesType
tnlGrid< 2, Real, Device, Index >::getFaceCoordinates( const Index faceIndex, int& nx, int& ny ) const
{
   tnlAssert( faceIndex >= 0 && faceIndex < ( this->template getEntitiesCount< Faces >() ),
              cerr << " faceIndex = " << faceIndex
                   << " this->template getEntitiesCount< Faces >() = " << ( this->template getEntitiesCount< Faces >() ) );
   if( faceIndex < this->numberOfNxFaces )
   {
      nx = 1;
      ny = 0;
      const IndexType aux = this->getDimensions().x() + 1;
      return CoordinatesType( faceIndex % aux, faceIndex / aux );
   }
   nx = 0;
   ny = 1;
   const IndexType i = faceIndex - this->numberOfNxFaces;
   const IndexType& aux = this->getDimensions().x();
   return CoordinatesType( i % aux, i / aux );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index tnlGrid< 2, Real, Device, Index > :: getVertexIndex( const CoordinatesType& vertexCoordinates ) const
{
   tnlAssert( vertexCoordinates.x() >= 0 && vertexCoordinates.x() < this->getDimensions().x() + 1,
              cerr << "vertexCoordinates.x() = " << vertexCoordinates.x()
                   << " this->getDimensions().x() + 1 = " << this->getDimensions().x() + 1 );
   tnlAssert( vertexCoordinates.y() >= 0 && vertexCoordinates.y() < this->getDimensions().y() + 1,
              cerr << "vertexCoordinates.y() = " << vertexCoordinates.y()
                   << " this->getDimensions().y() + 1 = " << this->getDimensions().y() + 1 );
   return vertexCoordinates.y() * ( this->dimensions.x() +1 ) + vertexCoordinates.x();
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
typename tnlGrid< 2, Real, Device, Index > :: CoordinatesType
tnlGrid< 2, Real, Device, Index > :: getVertexCoordinates( const Index vertexIndex ) const
{
   tnlAssert( vertexIndex >= 0 && vertexIndex < this->getNumberOfVertices(),
              cerr << " vertexIndex = " << vertexIndex
                   << " this->getNumberOfVertices() = " << this->getNumberOfVertices() );
   const IndexType aux = this->dimensions.x() + 1;
   return CoordinatesType( vertexIndex % aux, vertexIndex / aux );
}

template< typename Real,
          typename Device,
          typename Index >
   template< int dx, int dy >
__cuda_callable__
Index tnlGrid< 2, Real, Device, Index >::getCellNextToCell( const IndexType& cellIndex ) const
{
   const IndexType result = cellIndex + dx + dy * this->getDimensions().x();
   tnlAssert( result >= 0 &&
              result < this->template getEntitiesCount< 2 >(),
              cerr << " cellIndex = " << cellIndex
                   << " dx = " << dx
                   << " dy = " << dy
                   << " this->template getEntitiesCount< 2 >() = " << this->template getEntitiesCount< 2 >() );
   return result;
}

template< typename Real,
          typename Device,
          typename Index >
   template< int nx, int ny >
__cuda_callable__
Index tnlGrid< 2, Real, Device, Index >::getFaceNextToCell( const IndexType& cellIndex ) const
{
   tnlAssert( nx * ny == 0 && nx + ny != 0,
              cerr << "nx = " << nx
                   << "ny = " << ny );
   IndexType result;
   if( nx )
      result = cellIndex + cellIndex / this->getDimensions().x() + ( nx + ( nx < 0 ) );
   if( ny )
      result = this->numberOfNxFaces + cellIndex + ( ny + ( ny < 0 ) ) * this->getDimensions().x();
   tnlAssert( result >= 0 &&
              result < ( this->template getEntitiesCount< Faces >() ),
              cerr << " cellIndex = " << cellIndex
                   << " nx = " << nx
                   << " ny = " << ny
                   << " this->template getEntitiesCout< Faces >() = " << ( this->template getEntitiesCount< Faces >() ) );
   return result;
}

template< typename Real,
          typename Device,
          typename Index >
   template< int nx, int ny >
__cuda_callable__
Index tnlGrid< 2, Real, Device, Index >::getCellNextToFace( const IndexType& faceIndex ) const
{
   tnlAssert( abs( nx ) + abs( ny ) == 1,
              cerr << "nx = " << nx << " ny = " << ny );
#ifndef NDEBUG
   int _nx, _ny;
#endif   
   tnlAssert( ( nx + this->getFaceCoordinates( faceIndex, _nx, _ny ).x() >= 0 &&
                nx + this->getFaceCoordinates( faceIndex, _nx, _ny ).x() <= this->getDimensions().x() ),
              cerr << " nx = " << nx
                   << " this->getFaceCoordinates( faceIndex, _nx, _ny ).x() = " << this->getFaceCoordinates( faceIndex, _nx, _ny ).x()
                   << " this->getDimensions().x()  = " << this->getDimensions().x() );
   tnlAssert( ( ny + this->getFaceCoordinates( faceIndex, _nx, _ny ).y() >= 0 &&
                      ny + this->getFaceCoordinates( faceIndex, _nx, _ny ).y() <= this->getDimensions().y() ),
              cerr << " ny = " << ny
                   << " this->getFaceCoordinates( faceIndex, _nx, _ny ).y() = " << this->getFaceCoordinates( faceIndex, _nx, _ny ).y()
                   << " this->getDimensions().y()  = " << this->getDimensions().y() );

   IndexType result;
   if( nx )
      result = faceIndex + ( nx - ( nx > 0 ) ) - faceIndex / ( this->getDimensions().x() + 1 );
   if( ny )
      result = faceIndex - this->numberOfNxFaces + ( ny - ( ny > 0 ) ) * this->getDimensions().x();
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 2, Real, Device, Index > :: getHx() const
{
   return this->hx;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 2, Real, Device, Index > :: getHxSquare() const
{
   return this->hxSquare;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 2, Real, Device, Index > :: getHxInverse() const
{
   return this->hxInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 2, Real, Device, Index > :: getHxSquareInverse() const
{
   return this->hxSquareInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 2, Real, Device, Index > :: getHy() const
{
   return this->hy;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 2, Real, Device, Index > :: getHySquare() const
{
   return this->hySquare;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 2, Real, Device, Index > :: getHyInverse() const
{
   return this->hyInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 2, Real, Device, Index > :: getHySquareInverse() const
{
   return this->hySquareInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 2, Real, Device, Index > :: getHxHy() const
{
   return this->hxhy;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 2, Real, Device, Index > :: getHxHyInverse() const
{
   return this->hxhyInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Real tnlGrid< 2, Real, Device, Index > :: getSmallestSpaceStep() const
{
   return Min( this->hx, this->hy );
}

/*template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index tnlGrid< 2, Real, Device, Index > :: template getEntitiesCount< Cells >() const
{
   return this->numberOfCells;
};

template< typename Real,
          typename Device,
          typename Index >
   template< int nx,
             int ny >
__cuda_callable__
Index tnlGrid< 2, Real, Device, Index > :: getNumberOfFaces() const
{
   return nx * this->numberOfNxFaces + ny * this->numberOfNyFaces;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index tnlGrid< 2, Real, Device, Index > :: getNumberOfVertices() const
{
   return this->numberOfVertices;
}
*/


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool tnlGrid< 2, Real, Device, Index > :: isBoundaryCell( const CoordinatesType& cellCoordinates ) const
{
   tnlAssert( cellCoordinates.x() >= 0 && cellCoordinates.x() < this->getDimensions().x(),
              cerr << "cellCoordinates.x() = " << cellCoordinates.x()
                   << " this->getDimensions().x() = " << this->getDimensions().x() );
   tnlAssert( cellCoordinates.y() >= 0 && cellCoordinates.y() < this->getDimensions().y(),
              cerr << "cellCoordinates.y() = " << cellCoordinates.y()
                   << " this->getDimensions().y() = " << this->getDimensions().y() );

   if( cellCoordinates.x() == 0 || cellCoordinates.x() == this->getDimensions().x() - 1 ||
       cellCoordinates.y() == 0 || cellCoordinates.y() == this->getDimensions().y() - 1 )
      return true;
   return false;
}


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool
tnlGrid< 2, Real, Device, Index >::
isBoundaryCell( const IndexType& cellIndex ) const
{
   tnlAssert( cellIndex >= 0 && cellIndex < this->template getEntitiesCount< Cells >(),
              cerr << " cellIndex = " << cellIndex
                   << " this->template getEntitiesCount< Cells >() = " << this->template getEntitiesCount< Cells >() );
   return this->isBoundaryCell( this->template getEntity< Cells >( cellIndex ).getCoordinates() );
}


template< typename Real,
          typename Device,
          typename Index >
   template< int nx, int ny >
__cuda_callable__
bool tnlGrid< 2, Real, Device, Index > :: isBoundaryFace( const CoordinatesType& faceCoordinates ) const
{
   tnlStaticAssert( nx >= 0 && ny >= 0 && nx + ny == 1, "Wrong template parameters nx or ny." );
   if( nx )
   {
      tnlAssert( faceCoordinates.x() >= 0 && faceCoordinates.x() < this->getDimensions().x() + 1,
                 cerr << "faceCoordinates.x() = " << faceCoordinates.x()
                      << " this->getDimensions().x() + 1 = " << this->getDimensions().x() + 1 );
      tnlAssert( faceCoordinates.y() >= 0 && faceCoordinates.y() < this->getDimensions().y(),
                 cerr << "faceCoordinates.y() = " << faceCoordinates.y()
                      << " this->getDimensions().y() = " << this->getDimensions().y() );
      if( faceCoordinates.x() == 0 || faceCoordinates.x() == this->getDimensions().x() )
         return true;
      return false;
   }
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


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool tnlGrid< 2, Real, Device, Index > :: isBoundaryVertex( const CoordinatesType& vertexCoordinates ) const
{
   tnlAssert( vertexCoordinates.x() >= 0 && vertexCoordinates.x() < this->getDimensions().x() + 1,
              cerr << "vertexCoordinates.x() = " << vertexCoordinates.x()
                   << " this->getDimensions().x() + 1 = " << this->getDimensions().x() + 1 );
   tnlAssert( vertexCoordinates.y() >= 0 && vertexCoordinates.y() < this->getDimensions().y() + 1,
              cerr << "vertexCoordinates.y() = " << vertexCoordinates.y()
                   << " this->getDimensions().y() + 1 = " << this->getDimensions().y() + 1 );

   if( vertexCoordinates.x() == 0 || vertexCoordinates.x() == this->getDimensions().x() ||
       vertexCoordinates.y() == 0 || vertexCoordinates.y() == this->getDimensions().y() )
      return true;
   return false;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
      typename GridFunction::RealType
         tnlGrid< 2, Real, Device, Index >::getAbsMax( const GridFunction& f ) const
{
   return f.absMax();
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
      typename GridFunction::RealType
         tnlGrid< 2, Real, Device, Index >::getLpNorm( const GridFunction& f1,
                                                                 const typename GridFunction::RealType& p ) const
{
   typename GridFunction::RealType lpNorm( 0.0 );
   for( IndexType j = 0; j < getDimensions(). y(); j++ )
      for( IndexType i = 0; i < getDimensions(). x(); i++ )
      {
         IndexType c = this->getElementIndex( i, j );
         lpNorm += pow( tnlAbs( f1[ c ] ), p ) *
            this->getElementMeasure( CoordinatesType( i, j ) );
      }
   return pow( lpNorm, 1.0/p );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
      typename GridFunction::RealType
         tnlGrid< 2, Real, Device, Index >::getDifferenceAbsMax( const GridFunction& f1,
                                                                           const GridFunction& f2 ) const
{
   return f1.differenceAbsMax( f2 );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
      typename GridFunction::RealType
         tnlGrid< 2, Real, Device, Index >::getDifferenceLpNorm( const GridFunction& f1,
                                                                           const GridFunction& f2,
                                                                           const typename GridFunction::RealType& p ) const
{
   typename GridFunction::RealType lpNorm( 0.0 );
   for( IndexType j = 0; j < getDimensions(). y(); j++ )
      for( IndexType i = 0; i < getDimensions(). x(); i++ )
      {
         IndexType c = this->getCellIndex( CoordinatesType( i, j ) );
         lpNorm += pow( tnlAbs( f1[ c ] - f2[ c ] ), p );
      }
   lpNorm *= this->cellProportions.x() * this->cellProportions.y();
   return pow( lpNorm, 1.0 / p );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlGrid< 2, Real, Device, Index > :: save( tnlFile& file ) const
{
   if( ! tnlObject::save( file ) )
      return false;
   if( ! this->origin.save( file ) ||
       ! this->proportions.save( file ) ||
       ! this -> dimensions.save( file ) )
   {
      cerr << "I was not able to save the domain description of a tnlGrid." << endl;
      return false;
   }
   return true;
};

template< typename Real,
          typename Device,
          typename Index >
bool tnlGrid< 2, Real, Device, Index > :: load( tnlFile& file )
{
   if( ! tnlObject::load( file ) )
      return false;
   CoordinatesType dimensions;
   if( ! this->origin.load( file ) ||
       ! this->proportions.load( file ) ||
       ! dimensions.load( file ) )
   {
      cerr << "I was not able to load the domain description of a tnlGrid." << endl;
      return false;
   }
   this -> setDimensions( dimensions );
   return true;
};

template< typename Real,
          typename Device,
          typename Index >
bool tnlGrid< 2, Real, Device, Index > :: save( const tnlString& fileName ) const
{
   return tnlObject :: save( fileName );
};

template< typename Real,
           typename Device,
           typename Index >
bool tnlGrid< 2, Real, Device, Index > :: load( const tnlString& fileName )
{
   return tnlObject :: load( fileName );
};

template< typename Real,
          typename Device,
          typename Index >
bool tnlGrid< 2, Real, Device, Index > :: writeMesh( const tnlString& fileName,
                                                     const tnlString& format ) const
{
   fstream file;
   file. open( fileName. getString(), ios :: out );
   if( ! file )
   {
      cerr << "I am not able to open the file " << fileName << "." << endl;
      return false;
   }
   if( format == "asymptote" )
   {
      file << "size( "
           << this -> getProportions(). x() << "cm , "
           << this -> getProportions(). y() << "cm );"
           << endl << endl;
      GridEntity< 0 > vertex;
      CoordinatesType& vertexCoordinates = vertex.getCoordinates();
      VertexType v;
      for( Index j = 0; j < this -> dimensions. y(); j ++ )
      {
         file << "draw( ";
         vertexCoordinates.x() = 0;
         vertexCoordinates.y() = j;
         v = this->template getEntityCenter< 0, VertexType >( vertex );
         file << "( " << v. x() << ", " << v. y() << " )";
         for( Index i = 0; i < this -> dimensions. x(); i ++ )
         {
            vertexCoordinates.x() = i + 1;
            vertexCoordinates.y() = j;
            v = this -> getEntityCenter< 0, VertexType >( vertex );
            file << "--( " << v. x() << ", " << v. y() << " )";
         }
         file << " );" << endl;
      }
      file << endl;
      for( Index i = 0; i < this -> dimensions. x(); i ++ )
      {
         file << "draw( ";
         vertexCoordinates.x() = i;
         vertexCoordinates.y() = 0;
         v = this -> getEntityCenter< 0, VertexType >( vertex );
         file << "( " << v. x() << ", " << v. y() << " )";
         for( Index j = 0; j < this -> dimensions. y(); j ++ )
         {
            vertexCoordinates.x() = i;
            vertexCoordinates.y() = j + 1;
            v = this->template getEntityCenter< 0, VertexType >( vertex );
            file << "--( " << v. x() << ", " << v. y() << " )";
         }
         file << " );" << endl;
      }
      file << endl;

      GridEntity< 2 > cell;
      CoordinatesType& cellCoordinates = cell.getCoordinates();
      const RealType cellMeasure = this->cellProportions.x() * this->cellProportions.y();
      for( Index i = 0; i < this -> dimensions. x(); i ++ )
         for( Index j = 0; j < this -> dimensions. y(); j ++ )
         {
            cellCoordinates.x() = i;
            cellCoordinates.y() = j;
            v = this->template getEntityCenter< 2, VertexType >( cell );
            file << "label( scale(0.33) * Label( \"$" << setprecision( 3 ) << cellMeasure << setprecision( 8 )
                 << "$\" ), ( " << v. x() << ", " << v. y() << " ), S );" << endl;
         }

      for( Index i = 0; i < this -> dimensions. x(); i ++ )
         for( Index j = 0; j < this -> dimensions. y(); j ++ )
         {
            VertexType v1, v2, c;

            /****
             * East edge normal
             */
            /*v1 = this -> getVertex( CoordinatesType( i + 1, j ), v1 );
            v2 = this -> getVertex( CoordinatesType( i + 1, j + 1 ), v2 );
            c = ( ( Real ) 0.5 ) * ( v1 + v2 );
            this -> getEdgeNormal< 1, 0 >( CoordinatesType( i, j ), v );
            v *= 0.5;
            file << "draw( ( " << c. x() << ", " << c. y() << " )--( "
                 << c. x() + v. x() << ", " << c.y() + v. y() << " ), Arrow(size=1mm),p=green);" << endl;
            */
            /****
             * West edge normal
             */
            /*this -> getVertex< -1, -1 >( CoordinatesType( i, j ), v1 );
            this -> getVertex< -1, 1 >( CoordinatesType( i, j ), v2 );
            c = ( ( Real ) 0.5 ) * ( v1 + v2 );
            this -> getEdgeNormal< -1, 0 >( CoordinatesType( i, j ), v );
            v *= 0.5;
            file << "draw( ( " << c. x() << ", " << c. y() << " )--( "
                 << c. x() + v. x() << ", " << c.y() + v. y() << " ), Arrow(size=1mm),p=blue);" << endl;
            */
            /****
             * North edge normal
             */
            /*this -> getVertex< 1, 1 >( CoordinatesType( i, j ), v1 );
            this -> getVertex< -1, 1 >( CoordinatesType( i, j ), v2 );
            c = ( ( Real ) 0.5 ) * ( v1 + v2 );
            this -> getEdgeNormal< 0, 1 >( CoordinatesType( i, j ), v );
            v *= 0.5;
            file << "draw( ( " << c. x() << ", " << c. y() << " )--( "
                 << c. x() + v. x() << ", " << c.y() + v. y() << " ), Arrow(size=1mm),p=green);" << endl;
            */
            /****
             * South edge normal
             */
            /*this -> getVertex< 1, -1 >( CoordinatesType( i, j ), v1 );
            this -> getVertex< -1, -1 >( CoordinatesType( i, j ), v2 );
            c = ( ( Real ) 0.5 ) * ( v1 + v2 );
            this -> getEdgeNormal< 0, -1 >( CoordinatesType( i, j ), v );
            v *= 0.5;
            file << "draw( ( " << c. x() << ", " << c. y() << " )--( "
                 << c. x() + v. x() << ", " << c.y() + v. y() << " ), Arrow(size=1mm),p=blue);" << endl;
            */
         }
      return true;
   }
   return false;
}

template< typename Real,
           typename Device,
           typename Index >
   template< typename MeshFunction >
bool tnlGrid< 2, Real, Device, Index > :: write( const MeshFunction& function,
                                                 const tnlString& fileName,
                                                 const tnlString& format ) const
{
   if( this->template getEntitiesCount< Cells >() != function. getSize() )
   {
      cerr << "The size ( " << function. getSize() 
           << " ) of a mesh function does not agree with the DOFs ( " 
           << this->template getEntitiesCount< Cells >() << " ) of a mesh." << endl;
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
      GridEntity< 2 > cell;
      CoordinatesType& cellCoordinates = cell.getCoordinates();
      for( cellCoordinates.y() = 0; cellCoordinates.y() < getDimensions(). y(); cellCoordinates.y() ++ )
      {
         for( cellCoordinates.x() = 0; cellCoordinates.x() < getDimensions(). x(); cellCoordinates.x() ++ )
         {
            VertexType v = this->template getEntityCenter< 2, VertexType >( cell );
            tnlGnuplotWriter::write( file,  v );
            tnlGnuplotWriter::write( file,  function[ this->getEntityIndex( cell ) ] );
            file << endl;
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
tnlGrid< 2, Real, Device, Index >::
writeProlog( tnlLogger& logger )
{
   logger.writeParameter( "Dimensions:", Dimensions );
   logger.writeParameter( "Domain origin:", this->origin );
   logger.writeParameter( "Domain proportions:", this->proportions );
   logger.writeParameter( "Domain dimensions:", this->dimensions );
   logger.writeParameter( "Cell proportions:", this->cellProportions );
}


#endif /* TNLGRID2D_IMPL_H_ */
