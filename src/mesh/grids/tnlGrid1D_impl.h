/***************************************************************************
                          tnlGrid1D_impl.h  -  description
                             -------------------
    begin                : Jan 16, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <fstream>
#include <iomanip>
#include <core/tnlString.h>
#include <core/tnlAssert.h>
#include <mesh/tnlGnuplotWriter.h>
#include <mesh/grids/tnlGridEntityGetter_impl.h>
#include <mesh/grids/tnlNeighbourGridEntityGetter1D_impl.h>
#include <mesh/grids/tnlGrid1D.h>
#include <mesh/grids/tnlGridEntityMeasureGetter.h>

namespace TNL {

template< typename Real,
          typename Device,
          typename Index >
tnlGrid< 1, Real, Device, Index >::tnlGrid()
: numberOfCells( 0 ),
  numberOfVertices( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index  >
tnlString tnlGrid< 1, Real, Device, Index >::getType()
{
   return tnlString( "tnlGrid< " ) +
          tnlString( getMeshDimensions() ) + ", " +
          tnlString( ::getType< RealType >() ) + ", " +
          tnlString( Device::getDeviceType() ) + ", " +
          tnlString( ::getType< IndexType >() ) + " >";
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlGrid< 1, Real, Device, Index >::getTypeVirtual() const
{
   return this->getType();
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlGrid< 1, Real, Device, Index >::getSerializationType()
{
   return HostType::getType();
};

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlGrid< 1, Real, Device, Index >::getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 1, Real, Device, Index >::computeSpaceSteps()
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
void tnlGrid< 1, Real, Device, Index >::setDimensions( const CoordinatesType& dimensions )
{
   this->setDimensions( dimensions. x() );
}

template< typename Real,
          typename Device,
          typename Index  >
__cuda_callable__ inline
const typename tnlGrid< 1, Real, Device, Index >::CoordinatesType&
   tnlGrid< 1, Real, Device, Index >::getDimensions() const
{
   return this->dimensions;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 1, Real, Device, Index >::setDomain( const VertexType& origin,
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
const typename tnlGrid< 1, Real, Device, Index >::VertexType&
  tnlGrid< 1, Real, Device, Index >::getOrigin() const
{
   return this->origin;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const typename tnlGrid< 1, Real, Device, Index >::VertexType&
   tnlGrid< 1, Real, Device, Index >::getProportions() const
{
   return this->proportions;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityType >
__cuda_callable__  inline
Index
tnlGrid< 1, Real, Device, Index >::
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
tnlGrid< 1, Real, Device, Index >::
getEntity( const IndexType& entityIndex ) const
{
   static_assert( EntityType::entityDimensions <= 1 &&
                  EntityType::entityDimensions >= 0, "Wrong grid entity dimensions." );
 
   return tnlGridEntityGetter< ThisType, EntityType >::getEntity( *this, entityIndex );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityType >
__cuda_callable__ inline
Index
tnlGrid< 1, Real, Device, Index >::
getEntityIndex( const EntityType& entity ) const
{
   static_assert( EntityType::entityDimensions <= 1 &&
                  EntityType::entityDimensions >= 0, "Wrong grid entity dimensions." );
 
   return tnlGridEntityGetter< ThisType, EntityType >::getEntityIndex( *this, entity );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename EntityType >
__cuda_callable__
Real
tnlGrid< 1, Real, Device, Index >::
getEntityMeasure( const EntityType& entity ) const
{
   return tnlGridEntityMeasureGetter< ThisType, EntityType::getDimensions() >::getMeasure( *this, entity );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Real
tnlGrid< 1, Real, Device, Index >::
getCellMeasure() const
{
   return this->template getSpaceStepsProducts< 1 >();
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
typename tnlGrid< 1, Real, Device, Index >::VertexType
tnlGrid< 1, Real, Device, Index >::
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
tnlGrid< 1, Real, Device, Index >::
getSpaceStepsProducts() const
{
   tnlAssert( xPow >= -2 && xPow <= 2,
              cerr << " xPow = " << xPow );
   return this->spaceStepsProducts[ xPow + 2 ];
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
Real tnlGrid< 1, Real, Device, Index >::
getSmallestSpaceStep() const
{
   return this->spaceSteps.x();
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
 
   Cell cell( *this );
   for( cell.getCoordinates().x() = 0;
        cell.getCoordinates().x() < getDimensions().x();
        cell.getCoordinates().x()++ )
   {
      IndexType c = this->getEntityIndex( cell );
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
   FunctionRealType lpNorm( 0.0 ), cellVolume( this->getSpaceSteps().x() );

   Cell cell( *this );
   for( cell.getCoordinates().x() = 0;
        cell.getCoordinates().x() < getDimensions().x();
        cell.getCoordinates().x()++ )
   {
      IndexType c = this->getEntityIndex( cell );
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
bool tnlGrid< 1, Real, Device, Index >::load( tnlFile& file )
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
   this->setDimensions( dimensions );
   return true;
};

template< typename Real,
          typename Device,
          typename Index >
bool tnlGrid< 1, Real, Device, Index >::save( const tnlString& fileName ) const
{
   return tnlObject::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlGrid< 1, Real, Device, Index >::load( const tnlString& fileName )
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
bool tnlGrid< 1, Real, Device, Index >::write( const MeshFunction& function,
                                                 const tnlString& fileName,
                                                 const tnlString& format ) const
{
   if( this->template getEntitiesCount< Cell >() != function. getSize() )
   {
      cerr << "The size ( " << function. getSize()
           << " ) of the mesh function does not agree with the DOFs ( "
           << this->template getEntitiesCount< Cell >() << " ) of a mesh." << endl;
      return false;
   }
   fstream file;
   file. open( fileName. getString(), ios::out );
   if( ! file )
   {
      cerr << "I am not able to open the file " << fileName << "." << endl;
      return false;
   }
   file << setprecision( 12 );
   const RealType hx = getSpaceSteps(). x();
   if( format == "gnuplot" )
   {
      typename ThisType::template MeshEntity< getMeshDimensions() > entity( *this );
      for( entity.getCoordinates().x() = 0;
           entity.getCoordinates().x() < getDimensions(). x();
           entity.getCoordinates().x() ++ )
      {
         VertexType v = entity.getCenter();
         tnlGnuplotWriter::write( file,  v );
         tnlGnuplotWriter::write( file,  function[ this->getEntityIndex( entity ) ] );
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
   logger.writeParameter( "Dimensions:", getMeshDimensions() );
   logger.writeParameter( "Domain origin:", this->origin );
   logger.writeParameter( "Domain proportions:", this->proportions );
   logger.writeParameter( "Domain dimensions:", this->dimensions );
   logger.writeParameter( "Space steps:", this->getSpaceSteps() );
}

} // namespace TNL
