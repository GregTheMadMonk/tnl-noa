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
#include <mesh/grids/tnlGridEntityGetter_impl.h>
#include <mesh/grids/tnlNeighbourGridEntityGetter3D_impl.h>
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
__cuda_callable__ inline
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
__cuda_callable__ inline
const typename tnlGrid< 3, Real, Device, Index >::VertexType&
tnlGrid< 3, Real, Device, Index >::getOrigin() const
{
   return this->origin;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const typename tnlGrid< 3, Real, Device, Index > :: VertexType&
   tnlGrid< 3, Real, Device, Index > :: getProportions() const
{
	return this->proportions;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const typename tnlGrid< 3, Real, Device, Index > :: VertexType&
   tnlGrid< 3, Real, Device, Index > :: getCellProportions() const
{
   return this->cellProportions;
}

template< typename Real,
          typename Device,
          typename Index >
   template< int EntityDimensions >
__cuda_callable__  inline
Index
tnlGrid< 3, Real, Device, Index >:: 
getEntitiesCount() const
{
   static_assert( EntityDimensions <= 3 &&
                  EntityDimensions >= 0, "Wrong grid entity dimensions." );
   
   switch( EntityDimensions )
   {
      case 3:
         return this->numberOfCells;
      case 2:
         return this->numberOfFaces;         
      case 1:
         return this->numberOfEdges;                  
      case 0:
         return this->numberOfVertices;
   }            
}

template< typename Real,
          typename Device,
          typename Index >
   template< int EntityDimensions >
 __cuda_callable__ inline
typename tnlGrid< 3, Real, Device, Index >::template GridEntity< EntityDimensions >
tnlGrid< 3, Real, Device, Index >::
getEntity( const IndexType& entityIndex ) const
{
   static_assert( EntityDimensions <= 3 &&
                  EntityDimensions >= 0, "Wrong grid entity dimensions." );
   
   return tnlGridEntityGetter< ThisType, EntityDimensions >::getEntity( *this, entityIndex );
}

template< typename Real,
          typename Device,
          typename Index >
   template< int EntityDimensions >
__cuda_callable__ inline
Index
tnlGrid< 3, Real, Device, Index >::
getEntityIndex( const GridEntity< EntityDimensions >& entity ) const
{
   static_assert( EntityDimensions <= 3 &&
                  EntityDimensions >= 0, "Wrong grid entity dimensions." );
   
   return tnlGridEntityGetter< ThisType, EntityDimensions >::getEntityIndex( *this, entity );
}

template< typename Real,
          typename Device,
          typename Index >
   template< int EntityDimensions,
             typename Vertex >
__cuda_callable__ inline
Vertex tnlGrid< 3, Real, Device, Index > :: getEntityCenter( const GridEntity< EntityDimensions >& entity ) const
{
   static_assert( EntityDimensions <= 3 &&
                  EntityDimensions >= 0, "Wrong grid entity dimensions." );
   tnlAssert( entity.getCoordinates() >= CoordinatesType( 0, 0, 0 ) &&
              entity.getCoordinates() <= this->getDimensions() - entity.getBasis(),
                    cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                         << " this->getDimensions() = " << this->getDimensions()
                         << " entity.getBasis() = " << entity.getBasis() );
   return Vertex( this->origin.x() + ( entity.getCoordinates().x() + 0.5 * entity.getBasis().x() ) * this->cellProportions.x(),
                  this->origin.y() + ( entity.getCoordinates().y() + 0.5 * entity.getBasis().y() ) * this->cellProportions.y(),
                  this->origin.z() + ( entity.getCoordinates().z() + 0.5 * entity.getBasis().z() ) * this->cellProportions.z() );
}


/*template< typename Real,
          typename Device,
          typename Index >
   template< int dx, int dy, int dz >
__cuda_callable__ inline
Index tnlGrid< 3, Real, Device, Index > :: getCellNextToCell( const IndexType& cellIndex ) const
{
   tnlAssert( cellIndex + dx >= 0 &&
              cellIndex + dx < this->template getEntitiesCount< Cells >() &&
              cellIndex + dy * this->getDimensions().x() >= 0 &&
              cellIndex + dy * this->getDimensions().x() < this->template getEntitiesCount< Cells >() &&
              cellIndex + dz * this->cellZNeighboursStep >= 0 &&
              cellIndex + dz * this->cellZNeighboursStep < this->template getEntitiesCount< Cells >(),
              cerr << " cellIndex = " << cellIndex
                   << " dx = " << dx
                   << " dy = " << dy
                   << " dz = " << dz
                   << " this->template getEntitiesCount< Cells >() = " << this->template getEntitiesCount< Cells >() );
   return cellIndex + dx +
          dy * this->getDimensions().x() +
          dz * this->cellZNeighboursStep;
}

template< typename Real,
          typename Device,
          typename Index >
   template< int nx, int ny, int nz >
__cuda_callable__ inline
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
                   << " this->template getEntitiesCount< Cells >() = " << this->template getEntitiesCount< Cells >() );
   return result;
}

template< typename Real,
          typename Device,
          typename Index >
   template< int nx, int ny, int nz >
__cuda_callable__ inline
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
                   << " this->template getEntitiesCount< Cells >() = " << this->template getEntitiesCount< Cells >() );
   return result;
}*/

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const Real& tnlGrid< 3, Real, Device, Index > :: getHx() const
{
   return this->hx;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const Real& tnlGrid< 3, Real, Device, Index > :: getHxSquare() const
{
   return this->hxSquare;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const Real& tnlGrid< 3, Real, Device, Index > :: getHxInverse() const
{
   return this->hxInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const Real& tnlGrid< 3, Real, Device, Index > :: getHxSquareInverse() const
{
   return this->hxSquareInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const Real& tnlGrid< 3, Real, Device, Index > :: getHy() const
{
   return this->hy;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const Real& tnlGrid< 3, Real, Device, Index > :: getHySquare() const
{
   return this->hySquare;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const Real& tnlGrid< 3, Real, Device, Index > :: getHyInverse() const
{
   return this->hyInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const Real& tnlGrid< 3, Real, Device, Index > :: getHySquareInverse() const
{
   return this->hySquareInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const Real& tnlGrid< 3, Real, Device, Index > :: getHz() const
{
   return this->hz;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const Real& tnlGrid< 3, Real, Device, Index > :: getHzSquare() const
{
   return this->hzSquare;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const Real& tnlGrid< 3, Real, Device, Index > :: getHzInverse() const
{
   return this->hzInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const Real& tnlGrid< 3, Real, Device, Index > :: getHzSquareInverse() const
{
   return this->hzSquareInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const Real& tnlGrid< 3, Real, Device, Index > :: getHxHy() const
{
   return this->hxhy;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const Real& tnlGrid< 3, Real, Device, Index > :: getHxHz() const
{
   return this->hxhz;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const Real& tnlGrid< 3, Real, Device, Index > :: getHyHz() const
{
   return this->hyhz;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const Real& tnlGrid< 3, Real, Device, Index > :: getHxHyInverse() const
{
   return this->hxhyInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const Real& tnlGrid< 3, Real, Device, Index > :: getHxHzInverse() const
{
   return this->hxhzInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const Real& tnlGrid< 3, Real, Device, Index > :: getHyHzInverse() const
{
   return this->hyhzInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
Real tnlGrid< 3, Real, Device, Index > :: getSmallestSpaceStep() const
{
   return Min( this->hx, Min( this->hy, this->hz ) );
}

template< typename Real,
          typename Device,
          typename Index >
   template< int nx, int ny, int nz >
__cuda_callable__ inline
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
__cuda_callable__ inline
Index tnlGrid< 3, Real, Device, Index > :: getNumberOfEdges() const
{
   return dx * this->numberOfDxEdges +
          dy * this->numberOfDyEdges +
          dz * this->numberOfDzEdges;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
Index tnlGrid< 3, Real, Device, Index > :: getNumberOfVertices() const
{
   return numberOfVertices;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
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
__cuda_callable__ inline
bool
tnlGrid< 3, Real, Device, Index >::
isBoundaryCell( const IndexType& cellIndex ) const
{
   tnlAssert( cellIndex >= 0 && cellIndex < this->template getEntitiesCount< Cells >(),
              cerr << " cellIndex = " << cellIndex
                   << " this->template getEntitiesCount< Cells >() = " << this->template getEntitiesCount< Cells >() );
   //return this->isBoundaryCell( this->getCellCoordinates( cellIndex ) );
   return this->isBoundaryCell( this->template getEntity< Cells >( cellIndex ).getCoordinates() );
}

template< typename Real,
          typename Device,
          typename Index >
   template< int nx, int ny, int nz >
__cuda_callable__ inline
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
__cuda_callable__ inline
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
__cuda_callable__ inline
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
   GridEntity< Dimensions > cell;
   for( cell.getCoordinates().z() = 0;
        cell.getCoordinates().z() < getDimensions().z();
        cell.getCoordinates().z()++ )
      for( cell.getCoordinates().y() = 0;
           cell.getCoordinates().y() < getDimensions().y();
           cell.getCoordinates().y()++ )
         for( cell.getCoordinates().x() = 0;
              cell.getCoordinates().x() < getDimensions().x();
              cell.getCoordinates().z()++ )
         {
            IndexType c = this->getEntityIndex( cell );
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
   GridEntity< Dimensions > cell( *this );
   for( cell.getCoordinates().z() = 0;
        cell.getCoordinates().z() < getDimensions().z();
        cell.getCoordinates().z()++ )
      for( cell.getCoordinates().y() = 0;
           cell.getCoordinates().y() < getDimensions().y();
           cell.getCoordinates().y()++ )
         for( cell.getCoordinates().x() = 0;
              cell.getCoordinates().x() < getDimensions().x();
              cell.getCoordinates().z()++ )
         {
            IndexType c = this -> getEntityIndex( cell );
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
   GridEntity< Dimensions > cell( *this );

   for( cell.getCoordinates().z() = 0;
        cell.getCoordinates().z() < getDimensions().z();
        cell.getCoordinates().z()++ )
      for( cell.getCoordinates().y() = 0;
           cell.getCoordinates().y() < getDimensions().y();
           cell.getCoordinates().y()++ )
         for( cell.getCoordinates().x() = 0;
              cell.getCoordinates().x() < getDimensions().x();
              cell.getCoordinates().z()++ )
         {
            IndexType c = this->getEntityIndex( cell );
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
   if( this -> template getEntitiesCount< Cells >() != function. getSize() )
   {
      cerr << "The size ( " << function. getSize() 
           << " ) of a mesh function does not agree with the DOFs ( " << this -> template getEntitiesCount< Cells >() << " ) of a mesh." << endl;
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
      GridEntity< Cells > cell( *this );
      for( cell.getCoordinates().z() = 0;
           cell.getCoordinates().z() < getDimensions().z();
           cell.getCoordinates().z()++ )
      {
         for( cell.getCoordinates().y() = 0;
              cell.getCoordinates().y() < getDimensions().y();
              cell.getCoordinates().y()++ )
         {
            for( cell.getCoordinates().x() = 0;
                 cell.getCoordinates().x() < getDimensions().x();
                 cell.getCoordinates().x()++ )
            {
               VertexType v = this->template getEntityCenter< Cells, VertexType >( cell );
               tnlGnuplotWriter::write( file, v );
               tnlGnuplotWriter::write( file, function[ this->template getEntityIndex( cell ) ] );
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
