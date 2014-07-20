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

using namespace std;

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
tnlGrid< 1, Real, Device, Index, Geometry > :: tnlGrid()
: dofs( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry  >
tnlString tnlGrid< 1, Real, Device, Index, Geometry > :: getType()
{
   return tnlString( "tnlGrid< " ) +
          tnlString( Dimensions ) + ", " +
          tnlString( getParameterType< RealType >() ) + ", " +
          tnlString( Device :: getDeviceType() ) + ", " +
          tnlString( getParameterType< IndexType >() ) + ", " +
          Geometry< 1, Real, Device, Index > :: getType() + " >";
}

template< typename Real,
           typename Device,
           typename Index,
           template< int, typename, typename, typename > class Geometry >
tnlString tnlGrid< 1, Real, Device, Index, Geometry > :: getTypeVirtual() const
{
   return this -> getType();
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry  >
bool tnlGrid< 1, Real, Device, Index, Geometry > :: setDimensions( const Index xSize )
{
   tnlAssert( xSize > 0,
              cerr << "The number of Elements along x-axis must be larger than 0." );
   this -> dimensions. x() = xSize;
   dofs = xSize;

   VertexType parametricStep;
   parametricStep. x() = geometry. getProportions(). x() / xSize;
   geometry. setParametricStep( parametricStep );
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry  >
bool tnlGrid< 1, Real, Device, Index, Geometry > :: setDimensions( const CoordinatesType& dimensions )
{
   return this -> setDimensions( dimensions. x() );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry  >
const typename tnlGrid< 1, Real, Device, Index, Geometry > :: CoordinatesType& 
   tnlGrid< 1, Real, Device, Index, Geometry > :: getDimensions() const
{
   return this -> dimensions;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry  >
void tnlGrid< 1, Real, Device, Index, Geometry > :: setOrigin( const VertexType& origin )
{
   this -> origin = origin;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry  >
const typename tnlGrid< 1, Real, Device, Index, Geometry > :: VertexType& 
  tnlGrid< 1, Real, Device, Index, Geometry > :: getOrigin() const
{
   return this -> origin;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry  >
void tnlGrid< 1, Real, Device, Index, Geometry > :: setProportions( const VertexType& proportions )
{
   this->geometry.setProportions( proportions );
   this -> setDimensions( this -> dimensions );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry  >
const typename tnlGrid< 1, Real, Device, Index, Geometry > :: VertexType& 
   tnlGrid< 1, Real, Device, Index, Geometry > :: getProportions() const
{
   return this -> geometry.getProportions();
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry  >
void tnlGrid< 1, Real, Device, Index, Geometry > :: setParametricStep( const VertexType& parametricStep )
{
   VertexType v;
   v.x() = this -> dimensions. x() * parametricStep. x();
   this->geometry.setProportions( v );
   geometry. setParametricStep( parametricStep );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry  >
const typename tnlGrid< 1, Real, Device, Index, Geometry > :: VertexType& 
   tnlGrid< 1, Real, Device, Index, Geometry > :: getParametricStep() const
{
   return geometry. getParametricStep();
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
Index tnlGrid< 1, Real, Device, Index, Geometry > :: getElementIndex( const Index i ) const
{
   tnlAssert( i < dimensions. x(),
              cerr << "Index i ( " << i
                   << " ) is out of range ( " << dimensions. x()
                   << " ) in tnlGrid " << this -> getName(); )
   return i;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
void tnlGrid< 1, Real, Device, Index, Geometry > :: getElementCoordinates( const Index element,
                                                                           CoordinatesType& coordinates ) const
{
   tnlAssert( element >= 0 && element < dofs,
              cerr << " element = " << element << " dofs = " << dofs
                   << " in tnlGrid " << this -> getName(); );
   coordinates.x() = element;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
   template< typename Vertex >
void tnlGrid< 1, Real, Device, Index, Geometry > :: getElementCenter( const CoordinatesType& coordinates,
                                                                      Vertex& v ) const
{
   tnlAssert( coordinates.x() >= 0 && coordinates.x() < dofs,
              cerr << " element = " << coordinates.x() << " dofs = " << dofs
                   << " in tnlGrid " << this -> getName(); );
   return getVertex< 0 >( coordinates, v );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
Index tnlGrid< 1, Real, Device, Index, Geometry > :: getDofs() const
{
   return this -> dofs;
};

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
   template< int dx, typename Vertex >
void tnlGrid< 1, Real, Device, Index, Geometry > :: getVertex( const CoordinatesType& elementCoordinates,
                                                               Vertex& vertex ) const
{
   tnlAssert( elementCoordinates.x() >= 0 &&
              elementCoordinates.x() < this -> dimensions.x(),
              cerr << "elementCoordinates = " << elementCoordinates << endl; );
   vertex.x() = this->origin.x() + elementCoordinates.x() * getParametricStep().x();
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
Real tnlGrid< 1, Real, Device, Index, Geometry >::getElementMeasure( const CoordinatesType& coordinates ) const
{
   return getParametricStep().x();
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
   template< typename GridFunction >
      typename GridFunction::RealType
         tnlGrid< 1, Real, Device, Index, Geometry >::getDifferenceAbsMax( const GridFunction& f1,
                                                                           const GridFunction& f2 ) const
{
   typename GridFunction::RealType maxDiff( -1.0 );
   for( IndexType i = 0; i < getDimensions(). x(); i++ )
   {
      IndexType c = this -> getElementIndex( i );
      maxDiff = Max( maxDiff, tnlAbs( f1[ c ] - f2[ c ] ) );
   }
   return maxDiff;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
   template< typename GridFunction >
      typename GridFunction::RealType
         tnlGrid< 1, Real, Device, Index, Geometry >::getDifferenceLpNorm( const GridFunction& f1,
                                                                           const GridFunction& f2,
                                                                           const typename GridFunction::RealType& p ) const
{
   typename GridFunction::RealType lpNorm( 0.0 );
   for( IndexType i = 0; i < getDimensions(). x(); i++ )
   {
      IndexType c = this->getElementIndex( i );
      lpNorm += pow( tnlAbs( f1[ c ] - f2[ c ] ), p ) *
         this->getElementMeasure( CoordinatesType( i ) );
   }
   return pow( lpNorm, 1.0 / p );
}


template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
bool tnlGrid< 1, Real, Device, Index, Geometry >::save( tnlFile& file ) const
{
   if( ! tnlObject :: save( file ) )
      return false;
   if( ! this -> origin. save( file ) ||
       ! this -> dimensions. save( file ) )
   {
      cerr << "I was not able to save the domain description of the tnlGrid "
           << this -> getName() << endl;
      return false;
   }
   if( ! geometry. save( file ) )
   {
      cerr << "I was not able to save the mesh." << endl;
      return false;
   }
   return true;
};

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
bool tnlGrid< 1, Real, Device, Index, Geometry > :: load( tnlFile& file )
{
   if( ! tnlObject :: load( file ) )
      return false;
   CoordinatesType dim;
   if( ! this -> origin. load( file ) ||
       ! dim. load( file ) )
   {
      cerr << "I was not able to load the domain description of the tnlGrid "
           << this -> getName() << endl;
      return false;
   }
   if( ! geometry. load( file ) )
   {
      cerr << "I am not able to load the grid geometry." << endl;
      return false;
   }
   if( ! this -> setDimensions( dim ) )
   {
      cerr << "I am not able to allocate the loaded grid." << endl;
      return false;
   }
   //this -> refresh();
   return true;
};

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
bool tnlGrid< 1, Real, Device, Index, Geometry > :: save( const tnlString& fileName ) const
{
   return tnlObject::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
bool tnlGrid< 1, Real, Device, Index, Geometry > :: load( const tnlString& fileName )
{
   return tnlObject::load( fileName );
}

template< typename Real,
           typename Device,
           typename Index,
           template< int, typename, typename, typename > class Geometry >
bool tnlGrid< 1, Real, Device, Index, Geometry >::writeMesh( const tnlString& fileName,
                                                             const tnlString& format ) const
{
   return true;
}

template< typename Real,
           typename Device,
           typename Index,
           template< int, typename, typename, typename > class Geometry >
   template< typename MeshFunction >
bool tnlGrid< 1, Real, Device, Index, Geometry > :: write( const MeshFunction& function,
                                                           const tnlString& fileName,
                                                           const tnlString& format ) const
{
   if( this -> getDofs() != function. getSize() )
   {
      cerr << "The size ( " << function. getSize() << " ) of the mesh function " << function. getName()
           << " does not agree with the DOFs ( " << this -> getDofs() << " ) of the mesh " << this -> getName() << "." << endl;
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
   const RealType hx = getParametricStep(). x();
   if( format == "gnuplot" )
   {
      for( IndexType i = 0; i < getDimensions(). x(); i++ )
      {
         VertexType v;
         this -> getVertex< 0 >( CoordinatesType( i ), v );
         tnlGnuplotWriter::write( file,  v );
         tnlGnuplotWriter::write( file,  function[ this -> getElementIndex( i ) ] );
         file << endl;
      }
   }
   file. close();
   return true;
}

#endif /* TNLGRID1D_IMPL_H_ */
