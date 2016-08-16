/***************************************************************************
                          Grid1D_impl.h  -  description
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
#include <TNL/Meshes/GridDetails/NeighbourGridEntityGetter1D_impl.h>
#include <TNL/Meshes/GridDetails/Grid1D.h>
#include <TNL/Meshes/GridDetails/GridEntityMeasureGetter.h>

namespace TNL {
namespace Meshes {

template< typename Real,
          typename Device,
          typename Index >
Grid< 1, Real, Device, Index >::Grid()
: numberOfCells( 0 ),
  numberOfVertices( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index  >
String Grid< 1, Real, Device, Index >::getType()
{
   return String( "Meshes::Grid< " ) +
          String( getMeshDimensions() ) + ", " +
          String( TNL::getType< RealType >() ) + ", " +
          String( Device::getDeviceType() ) + ", " +
          String( TNL::getType< IndexType >() ) + " >";
}

template< typename Real,
          typename Device,
          typename Index >
String Grid< 1, Real, Device, Index >::getTypeVirtual() const
{
   return this->getType();
}

template< typename Real,
          typename Device,
          typename Index >
String Grid< 1, Real, Device, Index >::getSerializationType()
{
   return HostType::getType();
};

template< typename Real,
          typename Device,
          typename Index >
String Grid< 1, Real, Device, Index >::getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Real,
          typename Device,
          typename Index >
void Grid< 1, Real, Device, Index >::computeSpaceSteps()
{
   if( this->getDimensions().x() != 0 )
   {
      this->spaceSteps.x() = this->proportions.x() / ( Real )  this->getDimensions().x();
      const RealType& hx = this->spaceSteps.x();
      this->spaceStepsProducts[ 0 ] = 1.0 / ( hx * hx );
      this->spaceStepsProducts[ 1 ] = 1.0 / hx;
      this->spaceStepsProducts[ 2 ] = 1.0;
      this->spaceStepsProducts[ 3 ] = hx;
      this->spaceStepsProducts[ 4 ] = hx * hx;
   }
}

template< typename Real,
          typename Device,
          typename Index  >
void Grid< 1, Real, Device, Index >::setDimensions( const Index xSize )
{
   Assert( xSize > 0, std::cerr << "xSize = " << xSize );
   this->dimensions.x() = xSize;
   this->numberOfCells = xSize;
   this->numberOfVertices = xSize + 1;
   computeSpaceSteps();
}

template< typename Real,
          typename Device,
          typename Index  >
void Grid< 1, Real, Device, Index >::setDimensions( const CoordinatesType& dimensions )
{
   this->setDimensions( dimensions. x() );
}

template< typename Real,
          typename Device,
          typename Index  >
__cuda_callable__ inline
const typename Grid< 1, Real, Device, Index >::CoordinatesType&
   Grid< 1, Real, Device, Index >::getDimensions() const
{
   return this->dimensions;
}

template< typename Real,
          typename Device,
          typename Index >
void Grid< 1, Real, Device, Index >::setDomain( const VertexType& origin,
                                                     const VertexType& proportions )
{
   this->origin = origin;
   this->proportions = proportions;
   computeSpaceSteps();
}

template< typename Real,
          typename Device,
          typename Index  >
__cuda_callable__ inline
const typename Grid< 1, Real, Device, Index >::VertexType&
  Grid< 1, Real, Device, Index >::getOrigin() const
{
   return this->origin;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const typename Grid< 1, Real, Device, Index >::VertexType&
   Grid< 1, Real, Device, Index >::getProportions() const
{
   return this->proportions;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityType >
__cuda_callable__  inline
Index
Grid< 1, Real, Device, Index >::
getEntitiesCount() const
{
   static_assert( EntityType::entityDimensions <= 1 &&
                  EntityType::entityDimensions >= 0, "Wrong grid entity dimensions." );
 
   switch( EntityType::entityDimensions )
   {
      case 1:
         return this->numberOfCells;
      case 0:
         return this->numberOfVertices;
   }
   return -1;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityType >
 __cuda_callable__ inline
EntityType
Grid< 1, Real, Device, Index >::
getEntity( const IndexType& entityIndex ) const
{
   static_assert( EntityType::entityDimensions <= 1 &&
                  EntityType::entityDimensions >= 0, "Wrong grid entity dimensions." );
 
   return GridEntityGetter< ThisType, EntityType >::getEntity( *this, entityIndex );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityType >
__cuda_callable__ inline
Index
Grid< 1, Real, Device, Index >::
getEntityIndex( const EntityType& entity ) const
{
   static_assert( EntityType::entityDimensions <= 1 &&
                  EntityType::entityDimensions >= 0, "Wrong grid entity dimensions." );
 
   return GridEntityGetter< ThisType, EntityType >::getEntityIndex( *this, entity );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityType >
__cuda_callable__
Real
Grid< 1, Real, Device, Index >::
getEntityMeasure( const EntityType& entity ) const
{
   return GridEntityMeasureGetter< ThisType, EntityType::getDimensions() >::getMeasure( *this, entity );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Real
Grid< 1, Real, Device, Index >::
getCellMeasure() const
{
   return this->template getSpaceStepsProducts< 1 >();
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
typename Grid< 1, Real, Device, Index >::VertexType
Grid< 1, Real, Device, Index >::
getSpaceSteps() const
{
   return this->spaceSteps;
}

template< typename Real,
          typename Device,
          typename Index >
   template< int xPow >
__cuda_callable__ inline
const Real&
Grid< 1, Real, Device, Index >::
getSpaceStepsProducts() const
{
   Assert( xPow >= -2 && xPow <= 2,
              std::cerr << " xPow = " << xPow );
   return this->spaceStepsProducts[ xPow + 2 ];
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
Real Grid< 1, Real, Device, Index >::
getSmallestSpaceStep() const
{
   return this->spaceSteps.x();
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
typename GridFunction::RealType
Grid< 1, Real, Device, Index >::getDifferenceAbsMax( const GridFunction& f1,
                                                        const GridFunction& f2 ) const
{
   typename GridFunction::RealType maxDiff( -1.0 );
 
   Cell cell( *this );
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
Grid< 1, Real, Device, Index >::getDifferenceLpNorm( const GridFunction& f1,
                                                        const GridFunction& f2,
                                                        const typename GridFunction::RealType& p ) const
{
   typedef typename GridFunction::RealType FunctionRealType;
   FunctionRealType lpNorm( 0.0 ), cellVolume( this->getSpaceSteps().x() );

   Cell cell( *this );
   for( cell.getCoordinates().x() = 0;
        cell.getCoordinates().x() < getDimensions().x();
        cell.getCoordinates().x()++ )
   {
      IndexType c = this->getEntityIndex( cell );
      lpNorm += ::pow( abs( f1[ c ] - f2[ c ] ), p );
   }
   lpNorm *= cellVolume;
   return ::pow( lpNorm, 1.0 / p );
}

template< typename Real,
          typename Device,
          typename Index >
bool Grid< 1, Real, Device, Index >::save( File& file ) const
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
bool Grid< 1, Real, Device, Index >::load( File& file )
{
   if( ! Object::load( file ) )
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
bool Grid< 1, Real, Device, Index >::save( const String& fileName ) const
{
   return Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
bool Grid< 1, Real, Device, Index >::load( const String& fileName )
{
   return Object::load( fileName );
}

template< typename Real,
           typename Device,
           typename Index >
bool Grid< 1, Real, Device, Index >::writeMesh( const String& fileName,
                                                   const String& format ) const
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
bool Grid< 1, Real, Device, Index >::write( const MeshFunction& function,
                                                 const String& fileName,
                                                 const String& format ) const
{
   if( this->template getEntitiesCount< Cell >() != function. getSize() )
   {
      std::cerr << "The size ( " << function. getSize()
           << " ) of the mesh function does not agree with the DOFs ( "
           << this->template getEntitiesCount< Cell >() << " ) of a mesh." << std::endl;
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
   const RealType hx = getSpaceSteps(). x();
   if( format == "gnuplot" )
   {
      typename ThisType::template MeshEntity< getMeshDimensions() > entity( *this );
      for( entity.getCoordinates().x() = 0;
           entity.getCoordinates().x() < getDimensions(). x();
           entity.getCoordinates().x() ++ )
      {
         VertexType v = entity.getCenter();
         GnuplotWriter::write( file,  v );
         GnuplotWriter::write( file,  function[ this->getEntityIndex( entity ) ] );
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
Grid< 1, Real, Device, Index >::
writeProlog( Logger& logger )
{
   logger.writeParameter( "Dimensions:", getMeshDimensions() );
   logger.writeParameter( "Domain origin:", this->origin );
   logger.writeParameter( "Domain proportions:", this->proportions );
   logger.writeParameter( "Domain dimensions:", this->dimensions );
   logger.writeParameter( "Space steps:", this->getSpaceSteps() );
}

} // namespace Meshes
} // namespace TNL