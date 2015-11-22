/***************************************************************************
                          tnlGrid1D_impl.h  -  description
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

#ifndef TNLGRID1D_IMPL_H_
#define TNLGRID1D_IMPL_H_

#include <fstream>
#include <iomanip>
#include <core/tnlString.h>
#include <core/tnlAssert.h>
#include <mesh/tnlGnuplotWriter.h>
#include <mesh/grids/tnlGridEntityGetter_impl.h>
#include <mesh/grids/tnlGrid1D.h>

using namespace std;

template< typename Real,
          typename Device,
          typename Index >
tnlGrid< 1, Real, Device, Index > :: tnlGrid()
: numberOfCells( 0 ),
  numberOfVertices( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index  >
tnlString tnlGrid< 1, Real, Device, Index > :: getType()
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
tnlString tnlGrid< 1, Real, Device, Index > :: getTypeVirtual() const
{
   return this -> getType();
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlGrid< 1, Real, Device, Index > :: getSerializationType()
{
   return HostType::getType();
};

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlGrid< 1, Real, Device, Index > :: getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 1, Real, Device, Index > :: computeSpaceSteps()
{
   if( this->getDimensions().x() != 0 )
   {
      this->cellProportions = this->proportions.x() / ( Real ) this->getDimensions().x();
      this->hx = this->proportions.x() / ( Real )  this->getDimensions().x();
      this->hxSquare = this->hx * this->hx;
      this->hxInverse = 1.0 / this->hx;
      this->hxSquareInverse = this->hxInverse * this->hxInverse;
   }
}

template< typename Real,
          typename Device,
          typename Index  >
void tnlGrid< 1, Real, Device, Index >::setDimensions( const Index xSize )
{
   tnlAssert( xSize > 0, cerr << "xSize = " << xSize );
   this->dimensions.x() = xSize;
   this->numberOfCells = xSize;
   this->numberOfVertices = xSize + 1;
   computeSpaceSteps();
}

template< typename Real,
          typename Device,
          typename Index  >
void tnlGrid< 1, Real, Device, Index > :: setDimensions( const CoordinatesType& dimensions )
{
   this -> setDimensions( dimensions. x() );
}

template< typename Real,
          typename Device,
          typename Index  >
__cuda_callable__
const typename tnlGrid< 1, Real, Device, Index >::CoordinatesType&
   tnlGrid< 1, Real, Device, Index > :: getDimensions() const
{
   return this->dimensions;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 1, Real, Device, Index > :: setDomain( const VertexType& origin,
                                                     const VertexType& proportions )
{
   this->origin = origin;
   this->proportions = proportions;
   computeSpaceSteps();
}

template< typename Real,
          typename Device,
          typename Index  >
__cuda_callable__
const typename tnlGrid< 1, Real, Device, Index > :: VertexType& 
  tnlGrid< 1, Real, Device, Index > :: getOrigin() const
{
   return this->origin;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const typename tnlGrid< 1, Real, Device, Index > :: VertexType& 
   tnlGrid< 1, Real, Device, Index > :: getProportions() const
{
   return this->proportions;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const typename tnlGrid< 1, Real, Device, Index > :: VertexType& 
   tnlGrid< 1, Real, Device, Index > :: getCellProportions() const
{
   return this->cellProportions;
}

template< typename Real,
          typename Device,
          typename Index >
   template< int EntityDimensions >
 __cuda_callable__
typename tnlGrid< 1, Real, Device, Index >::template GridEntity< EntityDimensions >
tnlGrid< 1, Real, Device, Index >::
getEntity( const IndexType& entityIndex ) const
{
   static_assert( EntityDimensions <= 1 &&
                  EntityDimensions >= 0, "Wrong grid entity dimensions." );
   
   return tnlGridEntityGetter< ThisType, EntityDimensions >::getEntity( *this, entityIndex );
}

template< typename Real,
          typename Device,
          typename Index >
   template< int EntityDimensions >
__cuda_callable__
Index
tnlGrid< 1, Real, Device, Index >::
getEntityIndex( const GridEntity< EntityDimensions >& entity ) const
{
   static_assert( EntityDimensions <= 1 &&
                  EntityDimensions >= 0, "Wrong grid entity dimensions." );
   
   return tnlGridEntityGetter< ThisType, EntityDimensions >::getEntityIndex( *this, entity );
}

template< typename Real,
          typename Device,
          typename Index >
   template< int EntityDimensions,
             typename Vertex >
__cuda_callable__
Vertex tnlGrid< 1, Real, Device, Index > :: getEntityCenter( const GridEntity< EntityDimensions >& entity ) const
{
   static_assert( EntityDimensions <= 1 &&
                  EntityDimensions >= 0, "Wrong grid entity dimensions." );
   tnlAssert( entity.getCoordinates() >= CoordinatesType( 0 ) &&
              entity.getCoordinates() <= this->getDimensions() - entity.getBasis(),
                    cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                         << " this->getDimensions() = " << this->getDimensions()
                         << " entity.getBasis() = " << entity.getBasis() );
   return Vertex( this->origin.x() + ( entity.getCoordinates().x() + 0.5 * entity.getBasis().x() ) * this->cellProportions.x() );
}







template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index tnlGrid< 1, Real, Device, Index > :: getCellIndex( const CoordinatesType& cellCoordinates ) const
{
   tnlAssert( cellCoordinates.x() >= 0 && cellCoordinates.x() < this->getDimensions().x(),
              cerr << "cellCoordinates.x() = " << cellCoordinates.x()
                   << " this->getDimensions().x() = " << this->getDimensions().x() );
   return cellCoordinates.x();
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
typename tnlGrid< 1, Real, Device, Index > :: CoordinatesType
tnlGrid< 1, Real, Device, Index > :: getCellCoordinates( const Index cellIndex ) const
{
   tnlAssert( cellIndex >= 0 && cellIndex < this->getNumberOfCells(),
              cerr << " cellIndex = " << cellIndex
                   << " this->getNumberOfCells() = " << this->getNumberOfCells() );
   return CoordinatesType( cellIndex );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index tnlGrid< 1, Real, Device, Index > :: getVertexIndex( const CoordinatesType& vertexCoordinates ) const
{
   tnlAssert( vertexCoordinates.x() >= 0 && vertexCoordinates.x() < this->getDimensions().x() + 1,
              cerr << "vertexCoordinates.x() = " << vertexCoordinates.x()
                   << " this->getDimensions().x() + 1 = " << this->getDimensions().x() + 1 );
   return vertexCoordinates.x();
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
typename tnlGrid< 1, Real, Device, Index > :: CoordinatesType
tnlGrid< 1, Real, Device, Index > :: getVertexCoordinates( const Index vertexIndex ) const
{
   tnlAssert( vertexIndex >= 0 && vertexIndex < this->getNumberOfVertices(),
              cerr << " vertexIndex = " << vertexIndex
                   << " this->getNumberOfVertices() = " << this->getNumberOfVertices() );
   return CoordinatesType( vertexIndex );
}

template< typename Real,
          typename Device,
          typename Index >
   template< int dx >
__cuda_callable__
Index tnlGrid< 1, Real, Device, Index > :: getCellNextToCell( const IndexType& cellIndex ) const
{
   tnlAssert( cellIndex + dx >= 0 &&
              cellIndex + dx < this->getNumberOfCells(),
              cerr << " cellIndex = " << cellIndex
                   << " dx = " << dx
                   << " this->getNumberOfCells() = " << this->getNumberOfCells() );
   return cellIndex + dx;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 1, Real, Device, Index > :: getHx() const
{
   return this->hx;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 1, Real, Device, Index > :: getHxSquare() const
{
   return this->hxSquare;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 1, Real, Device, Index > :: getHxInverse() const
{
   return this->hxInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real& tnlGrid< 1, Real, Device, Index > :: getHxSquareInverse() const
{
   return this->hxSquareInverse;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Real tnlGrid< 1, Real, Device, Index > :: getSmallestSpaceStep() const
{
   return this->hx;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityTopology,
             typename Vertex >
__cuda_callable__
Vertex tnlGrid< 1, Real, Device, Index > :: getEntityCenter( const CoordinatesType& coordinates ) const
{
   static_assert( EntityTopology::entityDimensions <= 1 &&
                  EntityTopology::entityDimensions >= 0, "Wrong grid entity dimensions." );
   tnlAssert( coordinates.x() >= 0 && 
              coordinates.x() <= this->getDimensions().x() - EntityTopology::EntityProportions::i1,
         cerr << "coordinates.x() = " << coordinates.x()
              << " this->getDimensions().x() = " << this->getDimensions().x()
              << " EntityTopology::EntityProportions::i1 = " << EntityTopology::EntityProportions::i1 );
   return this->origin.x() + 
       ( coordinates.x() + 0.5 * EntityTopology::EntityProportions::i1 ) * this->cellProportions.x();
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityTopology,
             typename Vertex >
__cuda_callable__
Vertex tnlGrid< 1, Real, Device, Index > :: getEntityCenter( const IndexType& index ) const
{
   static_assert( EntityTopology::entityDimensions <= 1 &&
                  EntityTopology::entityDimensions >= 0, "Wrong grid entity dimensions." );
   
   // TODO: add assertions
   
   if( EntityTopology::entityDimensions == Dimensions )
      return this->getCellCenter( index );
   if( EntityTopology::entityDimensions == Dimensions - 1 )
      return this->template getVertex( index );
}


template< typename Real,
          typename Device,
          typename Index >
   template< typename Vertex >
__cuda_callable__
Vertex tnlGrid< 1, Real, Device, Index >::getCellCenter( const CoordinatesType& cellCoordinates ) const
{
   tnlAssert( cellCoordinates.x() >= 0 && cellCoordinates.x() < this->getDimensions().x(),
              cerr << "cellCoordinates.x() = " << cellCoordinates.x()
                   << " this->getDimensions().x() = " << this->getDimensions().x() );
   return this->origin.x() + ( cellCoordinates.x() + 0.5 ) * this->cellProportions.x();
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vertex >
__cuda_callable__
Vertex tnlGrid< 1, Real, Device, Index >::getCellCenter( const IndexType& cellIndex ) const
{
   tnlAssert( cellIndex >= 0 && cellIndex < this->getNumberOfCells(),
              cerr << " cellIndex = " << cellIndex
                   << " this->getNumberOfCells() = " << this->getNumberOfCells() );
   return this->getCellCenter< VertexType >( this->getCellCoordinates( cellIndex ) );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vertex >
__cuda_callable__
Vertex tnlGrid< 1, Real, Device, Index >::getVertex( const CoordinatesType& vertexCoordinates ) const
{
   tnlAssert( vertexCoordinates.x() >= 0 && vertexCoordinates.x() < this->getDimensions().x() + 1,
              cerr << "vertexCoordinates.x() = " << vertexCoordinates.x()
                   << " this->getDimensions().x() = " << this->getDimensions().x() );
   return Vertex( this->origin.x() + vertexCoordinates.x() * this->cellProportions.x() );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index tnlGrid< 1, Real, Device, Index > :: getNumberOfCells() const
{
   return this->numberOfCells;
};

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index tnlGrid< 1, Real, Device, Index > :: getNumberOfVertices() const
{
   return this->numberOfVertices;
};

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool tnlGrid< 1, Real, Device, Index > :: isBoundaryCell( const CoordinatesType& cellCoordinates ) const
{
   tnlAssert( cellCoordinates.x() >= 0 && cellCoordinates.x() < this->getDimensions().x(),
              cerr << "cellCoordinates.x() = " << cellCoordinates.x()
                   << " this->getDimensions().x() = " << this->getDimensions().x() );
   if( cellCoordinates.x() == 0 || cellCoordinates.x() == this->getDimensions().x() - 1 )
      return true;
   return false;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool
tnlGrid< 1, Real, Device, Index >::
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
__cuda_callable__
bool tnlGrid< 1, Real, Device, Index > :: isBoundaryVertex( const CoordinatesType& vertexCoordinates ) const
{
   tnlAssert( vertexCoordinates.x() >= 0 && vertexCoordinates.x() < this->getDimensions().x() + 1,
              cerr << "vertexCoordinates.x() = " << vertexCoordinates.x()
                   << " this->getDimensions().x() + 1 = " << this->getDimensions().x() + 1 );
   if( vertexCoordinates.x() == 0 || vertexCoordinates.x() == this->getDimensions().x() )
      return true;
   return false;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
typename GridFunction::RealType
tnlGrid< 1, Real, Device, Index >::getDifferenceAbsMax( const GridFunction& f1,
                                                        const GridFunction& f2 ) const
{
   typename GridFunction::RealType maxDiff( -1.0 );
   for( IndexType i = 0; i < getDimensions().x(); i++ )
   {
      IndexType c = this -> getCellIndex( i );
      maxDiff = Max( maxDiff, tnlAbs( f1[ c ] - f2[ c ] ) );
   }
   return maxDiff;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
typename GridFunction::RealType
tnlGrid< 1, Real, Device, Index >::getDifferenceLpNorm( const GridFunction& f1,
                                                        const GridFunction& f2,
                                                        const typename GridFunction::RealType& p ) const
{
   typedef typename GridFunction::RealType FunctionRealType;
   FunctionRealType lpNorm( 0.0 ), cellVolume( this->cellProportions.x() );
   for( IndexType i = 0; i < getDimensions(). x(); i++ )
   {
      IndexType c = this->getCellIndex( i );
      lpNorm += pow( tnlAbs( f1[ c ] - f2[ c ] ), p );
   }
   lpNorm *= cellVolume;
   return pow( lpNorm, 1.0 / p );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlGrid< 1, Real, Device, Index >::save( tnlFile& file ) const
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
bool tnlGrid< 1, Real, Device, Index > :: load( tnlFile& file )
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
bool tnlGrid< 1, Real, Device, Index > :: save( const tnlString& fileName ) const
{
   return tnlObject::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlGrid< 1, Real, Device, Index > :: load( const tnlString& fileName )
{
   return tnlObject::load( fileName );
}

template< typename Real,
           typename Device,
           typename Index >
bool tnlGrid< 1, Real, Device, Index >::writeMesh( const tnlString& fileName,
                                                   const tnlString& format ) const
{
   /*****
    * TODO: implement this
    */
   return true;
}

template< typename Real,
           typename Device,
           typename Index >
   template< typename MeshFunction >
bool tnlGrid< 1, Real, Device, Index > :: write( const MeshFunction& function,
                                                 const tnlString& fileName,
                                                 const tnlString& format ) const
{
   if( this->getNumberOfCells() != function. getSize() )
   {
      cerr << "The size ( " << function. getSize()
           << " ) of the mesh function does not agree with the DOFs ( " 
           << this -> getNumberOfCells() << " ) of a mesh." << endl;
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
   const RealType hx = getCellProportions(). x();
   if( format == "gnuplot" )
   {
      for( IndexType i = 0; i < getDimensions(). x(); i++ )
      {
         VertexType v = this->getCellCenter< VertexType >( CoordinatesType( i ) );
         tnlGnuplotWriter::write( file,  v );
         tnlGnuplotWriter::write( file,  function[ this->getCellIndex( i ) ] );
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
tnlGrid< 1, Real, Device, Index >::
writeProlog( tnlLogger& logger )
{
   logger.writeParameter( "Dimensions:", Dimensions );
   logger.writeParameter( "Domain origin:", this->origin );
   logger.writeParameter( "Domain proportions:", this->proportions );
   logger.writeParameter( "Domain dimensions:", this->dimensions );
   logger.writeParameter( "Cell proportions:", this->cellProportions );
}

#endif /* TNLGRID1D_IMPL_H_ */
