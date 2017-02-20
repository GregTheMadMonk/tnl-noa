/***************************************************************************
                          Grid3D_impl.h  -  description
                             -------------------
    begin                : Jan 16, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iomanip>
#include <TNL/Assert.h>
#include <TNL/Meshes/GridDetails/GridEntityGetter_impl.h>
#include <TNL/Meshes/GridDetails/NeighbourGridEntityGetter3D_impl.h>
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
  numberOfVertices( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index >
String Grid< 3, Real, Device, Index > :: getType()
{
   return String( "Meshes::Grid< " ) +
          String( meshDimension ) + ", " +
          String( TNL::getType< RealType >() ) + ", " +
          String( Device :: getDeviceType() ) + ", " +
          String( TNL::getType< IndexType >() ) + " >";
}

template< typename Real,
          typename Device,
          typename Index >
String Grid< 3, Real, Device, Index > :: getTypeVirtual() const
{
   return this->getType();
}

template< typename Real,
          typename Device,
          typename Index >
String Grid< 3, Real, Device, Index > :: getSerializationType()
{
   return HostType::getType();
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
}

template< typename Real,
          typename Device,
          typename Index >
void Grid< 3, Real, Device, Index > :: setDimensions( const Index xSize, const Index ySize, const Index zSize )
{
   TNL_ASSERT( xSize > 0, std::cerr << "xSize = " << xSize );
   TNL_ASSERT( ySize > 0, std::cerr << "ySize = " << ySize );
   TNL_ASSERT( zSize > 0, std::cerr << "zSize = " << zSize );

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
   template< typename EntityType >
__cuda_callable__  inline
Index
Grid< 3, Real, Device, Index >::
getEntitiesCount() const
{
   return getEntitiesCount< EntityType::getDimension() >();
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityType >
 __cuda_callable__ inline
EntityType
Grid< 3, Real, Device, Index >::
getEntity( const IndexType& entityIndex ) const
{
   static_assert( EntityType::entityDimension <= 3 &&
                  EntityType::entityDimension >= 0, "Wrong grid entity dimension." );
 
   return GridEntityGetter< ThisType, EntityType >::getEntity( *this, entityIndex );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityType >
__cuda_callable__ inline
Index
Grid< 3, Real, Device, Index >::
getEntityIndex( const EntityType& entity ) const
{
   static_assert( EntityType::entityDimension <= 3 &&
                  EntityType::entityDimension >= 0, "Wrong grid entity dimension." );
 
   return GridEntityGetter< ThisType, EntityType >::getEntityIndex( *this, entity );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityType >
__cuda_callable__
Real
Grid< 3, Real, Device, Index >::
getEntityMeasure( const EntityType& entity ) const
{
   return GridEntityMeasureGetter< ThisType, EntityType::getDimensions() >::getMeasure( *this, entity );
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
   TNL_ASSERT( xPow >= -2 && xPow <= 2,
              std::cerr << " xPow = " << xPow );
   TNL_ASSERT( yPow >= -2 && yPow <= 2,
              std::cerr << " yPow = " << yPow );
   TNL_ASSERT( zPow >= -2 && zPow <= 2,
              std::cerr << " zPow = " << zPow );

   return this->spaceStepsProducts[ xPow + 2 ][ yPow + 2 ][ zPow + 2 ];
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
   template< typename GridFunction >
typename GridFunction::RealType
   Grid< 3, Real, Device, Index >::getAbsMax( const GridFunction& f ) const
{
   return f.absMax();
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
typename GridFunction::RealType
   Grid< 3, Real, Device, Index >::getLpNorm( const GridFunction& f1,
                                                 const typename GridFunction::RealType& p ) const
{
   typename GridFunction::RealType lpNorm( 0.0 );
   MeshEntity< getMeshDimension() > cell;
   for( cell.getCoordinates().z() = 0;
        cell.getCoordinates().z() < getDimensions().z();
        cell.getCoordinates().z()++ )
      for( cell.getCoordinates().y() = 0;
           cell.getCoordinates().y() < getDimensions().y();
           cell.getCoordinates().y()++ )
         for( cell.getCoordinates().x() = 0;
              cell.getCoordinates().x() < getDimensions().x();
              cell.getCoordinates().x()++ )
         {
            IndexType c = this->getEntityIndex( cell );
            lpNorm += ::pow( abs( f1[ c ] ), p );;
         }
   lpNorm *= this->getSpaceSteps()().x() * this->getSpaceSteps()().y() * this->getSpaceSteps()().z();
   return ::pow( lpNorm, 1.0/p );
}


template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
      typename GridFunction::RealType
         Grid< 3, Real, Device, Index >::getDifferenceAbsMax( const GridFunction& f1,
                                                                           const GridFunction& f2 ) const
{
   typename GridFunction::RealType maxDiff( -1.0 );
   MeshEntity< getMeshDimension() > cell( *this );
   for( cell.getCoordinates().z() = 0;
        cell.getCoordinates().z() < getDimensions().z();
        cell.getCoordinates().z()++ )
      for( cell.getCoordinates().y() = 0;
           cell.getCoordinates().y() < getDimensions().y();
           cell.getCoordinates().y()++ )
         for( cell.getCoordinates().x() = 0;
              cell.getCoordinates().x() < getDimensions().x();
              cell.getCoordinates().x()++ )
         {
            IndexType c = this->getEntityIndex( cell );
            maxDiff = max( maxDiff, abs( f1[ c ] - f2[ c ] ) );
         }
   return maxDiff;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
      typename GridFunction::RealType
         Grid< 3, Real, Device, Index >::getDifferenceLpNorm( const GridFunction& f1,
                                                                 const GridFunction& f2,
                                                                 const typename GridFunction::RealType& p ) const
{
   typename GridFunction::RealType lpNorm( 0.0 );
   MeshEntity< getMeshDimension() > cell( *this );

   for( cell.getCoordinates().z() = 0;
        cell.getCoordinates().z() < getDimensions().z();
        cell.getCoordinates().z()++ )
      for( cell.getCoordinates().y() = 0;
           cell.getCoordinates().y() < getDimensions().y();
           cell.getCoordinates().y()++ )
         for( cell.getCoordinates().x() = 0;
              cell.getCoordinates().x() < getDimensions().x();
              cell.getCoordinates().x()++ )
         {
            IndexType c = this->getEntityIndex( cell );
            lpNorm += ::pow( abs( f1[ c ] - f2[ c ] ), p );
         }
   lpNorm *= this->getSpaceSteps().x() * this->getSpaceSteps().y() * this->getSpaceSteps().z();
   return ::pow( lpNorm, 1.0 / p );
}

template< typename Real,
          typename Device,
          typename Index >
bool Grid< 3, Real, Device, Index > :: save( File& file ) const
{
   if( ! Object::save( file ) )
      return false;
   if( ! this->origin.save( file ) ||
       ! this->proportions.save( file ) ||
       ! this->dimensions.save( file ) )
   {
      std::cerr << "I was not able to save the domain description of a Grid." << std::endl;
      return false;
   }
   return true;
};

template< typename Real,
          typename Device,
          typename Index >
bool Grid< 3, Real, Device, Index > :: load( File& file )
{
   if( ! Object :: load( file ) )
      return false;
   CoordinatesType dimensions;
   if( ! this->origin.load( file ) ||
       ! this->proportions.load( file ) ||
       ! dimensions.load( file ) )
   {
      std::cerr << "I was not able to load the domain description of a Grid." << std::endl;
      return false;
   }
   this->setDimensions( dimensions );
   return true;
};

template< typename Real,
          typename Device,
          typename Index >
bool Grid< 3, Real, Device, Index > :: save( const String& fileName ) const
{
   return Object :: save( fileName );
};

template< typename Real,
          typename Device,
          typename Index >
bool Grid< 3, Real, Device, Index > :: load( const String& fileName )
{
   return Object :: load( fileName );
};

template< typename Real,
           typename Device,
           typename Index >
bool Grid< 3, Real, Device, Index >::writeMesh( const String& fileName,
                                                   const String& format ) const
{
   TNL_ASSERT( false, std::cerr << "TODO: FIX THIS"); // TODO: FIX THIS
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename MeshFunction >
bool Grid< 3, Real, Device, Index > :: write( const MeshFunction& function,
                                                 const String& fileName,
                                                 const String& format ) const
{
   if( this->template getEntitiesCount< Cell >() != function. getSize() )
   {
      std::cerr << "The size ( " << function. getSize()
           << " ) of a mesh function does not agree with the DOFs ( " << this->template getEntitiesCount< Cell >() << " ) of a mesh." << std::endl;
      return false;
   }
   std::fstream file;
   file. open( fileName. getString(), std::ios::out );
   if( ! file )
   {
      std::cerr << "I am not able to open the file " << fileName << "." << std::endl;
      return false;
   }
   file << std::setprecision( 12 );
   if( format == "gnuplot" )
   {
      Cell cell( *this );
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
               PointType v = cell.getCenter();
               GnuplotWriter::write( file, v );
               GnuplotWriter::write( file, function[ this->template getEntityIndex( cell ) ] );
               file << std::endl;
            }
         }
         file << std::endl;
      }
   }

   file. close();
   return true;
}

template< typename Real,
           typename Device,
           typename Index >
void
Grid< 3, Real, Device, Index >::
writeProlog( Logger& logger )
{
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
