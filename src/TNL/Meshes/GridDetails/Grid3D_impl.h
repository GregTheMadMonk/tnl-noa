/***************************************************************************
                          Grid3D_impl.h  -  description
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
#include <TNL/Meshes/GridDetails/NeighborGridEntityGetter3D_impl.h>
#include <TNL/Meshes/GridDetails/Grid3D.h>
#include <TNL/Meshes/GridDetails/GridEntityMeasureGetter.h>

namespace TNL {
namespace Meshes {

template< typename Real,
          typename Device,
          typename Index >
Grid< 3, Real, Device, Index > :: Grid()
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
  numberOfVertices( 0 ),
  distGrid(nullptr)
{
}

template< typename Real,
          typename Device,
          typename Index >
Grid< 3, Real, Device, Index >::Grid( const Index xSize, const Index ySize, const Index zSize )
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
  numberOfVertices( 0 ),
  distGrid(nullptr)
{
   this->setDimensions( xSize, ySize, zSize );
}


template< typename Real,
          typename Device,
          typename Index >
String Grid< 3, Real, Device, Index > :: getSerializationType()
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
String Grid< 3, Real, Device, Index > :: getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Real,
          typename Device,
          typename Index >
void Grid< 3, Real, Device, Index > :: computeSpaceSteps()
{
   if( this->getDimensions().x() > 0 &&
       this->getDimensions().y() > 0 &&
       this->getDimensions().z() > 0 )
   {
      this->spaceSteps.x() = this->proportions.x() / ( Real ) this->getDimensions().x();
      this->spaceSteps.y() = this->proportions.y() / ( Real ) this->getDimensions().y();
      this->spaceSteps.z() = this->proportions.z() / ( Real ) this->getDimensions().z();
      
      this->computeSpaceStepPowers();
      
   }
};

template< typename Real,
          typename Device,
          typename Index >
void Grid< 3, Real, Device, Index > :: computeSpaceStepPowers()
{      
      const RealType& hx = this->spaceSteps.x();
      const RealType& hy = this->spaceSteps.y();
      const RealType& hz = this->spaceSteps.z();

      Real auxX, auxY, auxZ;
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
            for( int k = 0; k < 5; k++ )
            {
               switch( k )
               {
                  case 0:
                     auxZ = 1.0 / ( hz * hz );
                     break;
                  case 1:
                     auxZ = 1.0 / hz;
                     break;
                  case 2:
                     auxZ = 1.0;
                     break;
                  case 3:
                     auxZ = hz;
                     break;
                  case 4:
                     auxZ = hz * hz;
                     break;
               }
               this->spaceStepsProducts[ i ][ j ][ k ] = auxX * auxY * auxZ;
            }
         }
      }
}


template< typename Real,
          typename Device,
          typename Index >
void Grid< 3, Real, Device, Index > :: computeProportions()
{
    this->proportions.x()=this->dimensions.x()*this->spaceSteps.x();
    this->proportions.y()=this->dimensions.y()*this->spaceSteps.y();
    this->proportions.z()=this->dimensions.z()*this->spaceSteps.z();
}

template< typename Real,
          typename Device,
          typename Index >
void Grid< 3, Real, Device, Index > :: setOrigin( const PointType& origin)
{
    this->origin=origin;
}

template< typename Real,
          typename Device,
          typename Index >
void Grid< 3, Real, Device, Index > :: setSpaceSteps(const PointType& steps)
{
     this->spaceSteps=steps;    
     computeSpaceStepPowers();
     computeProportions(); 
}


template< typename Real,
          typename Device,
          typename Index >
void Grid< 3, Real, Device, Index > :: setDimensions( const Index xSize, const Index ySize, const Index zSize )
{
   TNL_ASSERT_GT( xSize, 0, "Grid size must be positive." );
   TNL_ASSERT_GT( ySize, 0, "Grid size must be positive." );
   TNL_ASSERT_GT( zSize, 0, "Grid size must be positive." );

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

   this->cellZNeighborsStep = xSize * ySize;

   computeSpaceSteps();
}

template< typename Real,
          typename Device,
          typename Index >
void Grid< 3, Real, Device, Index > :: setDimensions( const CoordinatesType& dimensions )
{
   return this->setDimensions( dimensions. x(), dimensions. y(), dimensions. z() );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const typename Grid< 3, Real, Device, Index > :: CoordinatesType&
   Grid< 3, Real, Device, Index > :: getDimensions() const
{
   return this->dimensions;
}

template< typename Real,
          typename Device,
          typename Index >
void Grid< 3, Real, Device, Index > :: setDomain( const PointType& origin,
                                                     const PointType& proportions )
{
   this->origin = origin;
   this->proportions = proportions;
   computeSpaceSteps();
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const typename Grid< 3, Real, Device, Index >::PointType&
Grid< 3, Real, Device, Index >::getOrigin() const
{
   return this->origin;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const typename Grid< 3, Real, Device, Index > :: PointType&
   Grid< 3, Real, Device, Index > :: getProportions() const
{
	return this->proportions;
}


template< typename Real,
          typename Device,
          typename Index >
   template< int EntityDimension >
__cuda_callable__  inline
Index
Grid< 3, Real, Device, Index >::
getEntitiesCount() const
{
   static_assert( EntityDimension <= 3 &&
                  EntityDimension >= 0, "Wrong grid entity dimensions." );

   switch( EntityDimension )
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
   return -1;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Entity >
__cuda_callable__  inline
Index
Grid< 3, Real, Device, Index >::
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
Grid< 3, Real, Device, Index >::
getEntity( const IndexType& entityIndex ) const
{
   static_assert( Entity::getEntityDimension() <= 3 &&
                  Entity::getEntityDimension() >= 0, "Wrong grid entity dimensions." );

   return GridEntityGetter< Grid, Entity >::getEntity( *this, entityIndex );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Entity >
__cuda_callable__ inline
Index
Grid< 3, Real, Device, Index >::
getEntityIndex( const Entity& entity ) const
{
   static_assert( Entity::getEntityDimension() <= 3 &&
                  Entity::getEntityDimension() >= 0, "Wrong grid entity dimensions." );

   return GridEntityGetter< Grid, Entity >::getEntityIndex( *this, entity );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const typename Grid< 3, Real, Device, Index >::PointType&
Grid< 3, Real, Device, Index >::
getSpaceSteps() const
{
   return this->spaceSteps;
}

template< typename Real,
          typename Device,
          typename Index >
   template< int xPow, int yPow, int zPow >
__cuda_callable__ inline
const Real&
Grid< 3, Real, Device, Index >::
getSpaceStepsProducts() const
{
   static_assert( xPow >= -2 && xPow <= 2, "unsupported value of xPow" );
   static_assert( yPow >= -2 && yPow <= 2, "unsupported value of yPow" );
   static_assert( zPow >= -2 && zPow <= 2, "unsupported value of zPow" );
   return this->spaceStepsProducts[ xPow + 2 ][ yPow + 2 ][ zPow + 2 ];
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real&
Grid< 3, Real, Device, Index >::
getCellMeasure() const
{
   return this->template getSpaceStepsProducts< 1, 1, 1 >();
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
Real Grid< 3, Real, Device, Index > :: getSmallestSpaceStep() const
{
   return min( this->spaceSteps.x(), min( this->spaceSteps.y(), this->spaceSteps.z() ) );
}

template< typename Real,
          typename Device,
          typename Index >
void Grid< 3, Real, Device, Index >:: setDistMesh(DistributedMeshType * distMesh)
{
    this->distGrid=distMesh;
}
   
template< typename Real,
          typename Device,
          typename Index >
DistributedMeshes::DistributedMesh <Grid< 3, Real, Device, Index >> * 
Grid< 3, Real, Device, Index >:: getDistributedMesh(void) const
{
    return this->distGrid;
}

template< typename Real,
          typename Device,
          typename Index >
void Grid< 3, Real, Device, Index > :: save( File& file ) const
{
   Object::save( file );
   this->origin.save( file );
   this->proportions.save( file );
   this->dimensions.save( file );
};

template< typename Real,
          typename Device,
          typename Index >
void Grid< 3, Real, Device, Index >::load( File& file )
{
   Object :: load( file );
   CoordinatesType dimensions;
   this->origin.load( file );
   this->proportions.load( file );
   dimensions.load( file );
   this->setDimensions( dimensions );
};

template< typename Real,
          typename Device,
          typename Index >
void Grid< 3, Real, Device, Index >::save( const String& fileName ) const
{
   Object::save( fileName );
};

template< typename Real,
          typename Device,
          typename Index >
void Grid< 3, Real, Device, Index >::load( const String& fileName )
{
   Object::load( fileName );
};

template< typename Real,
          typename Device,
          typename Index >
void
Grid< 3, Real, Device, Index >::
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
