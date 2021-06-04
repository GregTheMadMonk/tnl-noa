/***************************************************************************
                          Grid2D_impl.h  -  description
                             -------------------
    begin                : Jan 16, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <fstream>
#include <iomanip>
#include <TNL/String.h>
#include <TNL/Assert.h>
#include <TNL/Meshes/GridDetails/GnuplotWriter.h>
#include <TNL/Meshes/GridDetails/GridEntityGetter_impl.h>
#include <TNL/Meshes/GridDetails/NeighborGridEntityGetter2D_impl.h>
#include <TNL/Meshes/GridDetails/Grid2D.h>
#include <TNL/Meshes/GridDetails/GridEntityMeasureGetter.h>

namespace TNL {
namespace Meshes {

template< typename Real,
          typename Device,
          typename Index >
Grid< 2, Real, Device, Index > :: Grid()
: numberOfCells( 0 ),
  numberOfNxFaces( 0 ),
  numberOfNyFaces( 0 ),
  numberOfFaces( 0 ),
  numberOfVertices( 0 ),
  distGrid(nullptr)
{
}

template< typename Real,
          typename Device,
          typename Index >
Grid< 2, Real, Device, Index >::Grid( const Index xSize, const Index ySize )
: numberOfCells( 0 ),
  numberOfNxFaces( 0 ),
  numberOfNyFaces( 0 ),
  numberOfFaces( 0 ),
  numberOfVertices( 0 ),
  distGrid(nullptr)
{
   this->setDimensions( xSize, ySize );
}

template< typename Real,
          typename Device,
          typename Index >
String Grid< 2, Real, Device, Index > :: getSerializationType()
{
   return String( "Meshes::Grid< " ) +
          convertToString( getMeshDimension() ) + ", " +
          getType< RealType >() + ", " +
          getType< Devices::Host >() + ", " +
          getType< IndexType >() + " >";
};

template< typename Real,
          typename Device,
          typename Index >
String Grid< 2, Real, Device, Index > :: getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
void Grid< 2, Real, Device, Index > :: computeSpaceSteps()
{
   if( this->getDimensions().x() > 0 && this->getDimensions().y() > 0 )
   {
      this->spaceSteps.x() = this->proportions.x() / ( Real ) this->getDimensions().x();
      this->spaceSteps.y() = this->proportions.y() / ( Real ) this->getDimensions().y();
      this->computeSpaceStepPowers();
   }
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
void Grid< 2, Real, Device, Index > ::computeSpaceStepPowers()
{
      const RealType& hx = this->spaceSteps.x();
      const RealType& hy = this->spaceSteps.y();

      Real auxX, auxY;
      for( int i = 0; i < 5; i++ )
      {
         switch( i )
         {
            case 0:
               auxX = 1.0 / ( hx * hx );
               break;
            case 1:
               auxX = 1.0 / hx;
               break;
            case 2:
               auxX = 1.0;
               break;
            case 3:
               auxX = hx;
               break;
            case 4:
               auxX = hx * hx;
               break;
         }
         for( int j = 0; j < 5; j++ )
         {
            switch( j )
            {
               case 0:
                  auxY = 1.0 / ( hy * hy );
                  break;
               case 1:
                  auxY = 1.0 / hy;
                  break;
               case 2:
                  auxY = 1.0;
                  break;
               case 3:
                  auxY = hy;
                  break;
               case 4:
                  auxY = hy * hy;
                  break;
            }
            this->spaceStepsProducts[ i ][ j ] = auxX * auxY;
         }
      }
}

template< typename Real,
          typename Device,
          typename Index >
void Grid< 2, Real, Device, Index > ::computeProportions()
{
    this->proportions.x()=this->dimensions.x()*this->spaceSteps.x();
    this->proportions.y()=this->dimensions.y()*this->spaceSteps.y();
}

template< typename Real,
          typename Device,
          typename Index >
void Grid< 2, Real, Device, Index > :: setDimensions( const Index xSize, const Index ySize )
{
   TNL_ASSERT_GT( xSize, 0, "Grid size must be positive." );
   TNL_ASSERT_GT( ySize, 0, "Grid size must be positive." );

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
void Grid< 2, Real, Device, Index > :: setDimensions( const CoordinatesType& dimensions )
{
   return this->setDimensions( dimensions. x(), dimensions. y() );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const typename Grid< 2, Real, Device, Index >::CoordinatesType&
Grid< 2, Real, Device, Index > :: getDimensions() const
{
   return this->dimensions;
}

template< typename Real,
          typename Device,
          typename Index >
void Grid< 2, Real, Device, Index > :: setDomain( const PointType& origin,
                                                     const PointType& proportions )
{
   this->origin = origin;
   this->proportions = proportions;
   computeSpaceSteps();
}

template< typename Real,
          typename Device,
          typename Index >
void Grid< 2, Real, Device, Index > :: setOrigin( const PointType& origin)
{
   this->origin = origin;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const typename Grid< 2, Real, Device, Index >::PointType&
Grid< 2, Real, Device, Index >::getOrigin() const
{
   return this->origin;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const typename Grid< 2, Real, Device, Index > :: PointType&
   Grid< 2, Real, Device, Index > :: getProportions() const
{
   return this->proportions;
}


template< typename Real,
          typename Device,
          typename Index >
   template< int EntityDimension >
__cuda_callable__ inline
Index
Grid< 2, Real, Device, Index >::
getEntitiesCount() const
{
   static_assert( EntityDimension <= 2 &&
                  EntityDimension >= 0, "Wrong grid entity dimensions." );

   switch( EntityDimension )
   {
      case 2:
         return this->numberOfCells;
      case 1:
         return this->numberOfFaces;
      case 0:
         return this->numberOfVertices;
   }
   return -1;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Entity >
__cuda_callable__ inline
Index
Grid< 2, Real, Device, Index >::
getEntitiesCount() const
{
   return getEntitiesCount< Entity::getEntityDimension() >();
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Entity >
__cuda_callable__ inline
Entity
Grid< 2, Real, Device, Index >::
getEntity( const IndexType& entityIndex ) const
{
   static_assert( Entity::getEntityDimension() <= 2 &&
                  Entity::getEntityDimension() >= 0, "Wrong grid entity dimensions." );

   return GridEntityGetter< Grid, Entity >::getEntity( *this, entityIndex );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Entity >
__cuda_callable__ inline
Index
Grid< 2, Real, Device, Index >::
getEntityIndex( const Entity& entity ) const
{
   static_assert( Entity::getEntityDimension() <= 2 &&
                  Entity::getEntityDimension() >= 0, "Wrong grid entity dimensions." );

   return GridEntityGetter< Grid, Entity >::getEntityIndex( *this, entity );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const typename Grid< 2, Real, Device, Index >::PointType&
Grid< 2, Real, Device, Index >::
getSpaceSteps() const
{
   return this->spaceSteps;
}

template< typename Real,
          typename Device,
          typename Index >
inline void 
Grid< 2, Real, Device, Index >::
setSpaceSteps(const typename Grid< 2, Real, Device, Index >::PointType& steps)
{
    this->spaceSteps=steps;
    computeSpaceStepPowers();
    computeProportions();
}

template< typename Real,
          typename Device,
          typename Index >
   template< int xPow, int yPow  >
__cuda_callable__ inline
const Real&
Grid< 2, Real, Device, Index >::
getSpaceStepsProducts() const
{
   static_assert( xPow >= -2 && xPow <= 2, "unsupported value of xPow" );
   static_assert( yPow >= -2 && yPow <= 2, "unsupported value of yPow" );
   return this->spaceStepsProducts[ xPow + 2 ][ yPow + 2 ];
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index
Grid< 2, Real, Device, Index >::
getNumberOfNxFaces() const
{
   return numberOfNxFaces;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real&
Grid< 2, Real, Device, Index >::
getCellMeasure() const
{
   return this->template getSpaceStepsProducts< 1, 1 >();
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
Real Grid< 2, Real, Device, Index > :: getSmallestSpaceStep() const
{
   return min( this->spaceSteps.x(), this->spaceSteps.y() );
}

template< typename Real,
          typename Device,
          typename Index >
void Grid< 2, Real, Device, Index >:: setDistMesh(DistributedMeshType* distMesh)
{
    this->distGrid=distMesh;
}
   
template< typename Real,
          typename Device,
          typename Index >
DistributedMeshes::DistributedMesh <Grid< 2, Real, Device, Index >> * 
Grid< 2, Real, Device, Index >:: getDistributedMesh(void) const
{
    return this->distGrid;
}

template< typename Real,
          typename Device,
          typename Index >
void Grid< 2, Real, Device, Index > :: save( File& file ) const
{
   Object::save( file );
   this->origin.save( file );
   this->proportions.save( file );
   this->dimensions.save( file );
};

template< typename Real,
          typename Device,
          typename Index >
void Grid< 2, Real, Device, Index > :: load( File& file )
{
   Object::load( file );
   CoordinatesType dimensions;
   this->origin.load( file );
   this->proportions.load( file );
   dimensions.load( file );
   this->setDimensions( dimensions );
};

template< typename Real,
          typename Device,
          typename Index >
void Grid< 2, Real, Device, Index > :: save( const String& fileName ) const
{
   Object::save( fileName );
};

template< typename Real,
           typename Device,
           typename Index >
void Grid< 2, Real, Device, Index > :: load( const String& fileName )
{
   Object::load( fileName );
};

template< typename Real,
          typename Device,
          typename Index >
void
Grid< 2, Real, Device, Index >::
writeProlog( Logger& logger ) const
{
   if( this->getDistributedMesh() && this->getDistributedMesh()->isDistributed() )
      return this->getDistributedMesh()->writeProlog( logger );
   logger.writeParameter( "Dimension:", getMeshDimension() );
   logger.writeParameter( "Domain origin:", this->origin );
   logger.writeParameter( "Domain proportions:", this->proportions );
   logger.writeParameter( "Domain dimensions:", this->dimensions );
   logger.writeParameter( "Space steps:", this->getSpaceSteps() );
   logger.writeParameter( "Number of cells:", getEntitiesCount< Cell >() );
   logger.writeParameter( "Number of faces:", getEntitiesCount< Face >() );
   logger.writeParameter( "Number of vertices:", getEntitiesCount< Vertex >() );
}

} // namespace Meshes
} // namespace TNL
