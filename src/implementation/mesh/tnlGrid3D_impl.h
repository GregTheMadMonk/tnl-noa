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

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
tnlGrid< 3, Real, Device, Index, Geometry > :: tnlGrid()
: dofs( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
tnlString tnlGrid< 3, Real, Device, Index, Geometry > :: getType()
{
   return tnlString( "tnlGrid< " ) +
          tnlString( Dimensions ) + ", " +
          tnlString( getParameterType< RealType >() ) + ", " +
          tnlString( Device :: getDeviceType() ) + ", " +
          tnlString( getParameterType< IndexType >() ) + ", " +
          Geometry< 3, Real, Device, Index > :: getType() + " >";
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
tnlString tnlGrid< 3, Real, Device, Index, Geometry > :: getTypeVirtual() const
{
   return this -> getType();
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
bool tnlGrid< 3, Real, Device, Index, Geometry > :: setDimensions( const Index xSize, const Index ySize, const Index zSize )
{
   tnlAssert( xSize > 1,
              cerr << "The number of Elements along x-axis must be larger than 1." );
   tnlAssert( ySize > 1,
              cerr << "The number of Elements along y-axis must be larger than 1." );
   tnlAssert( zSize > 1,
              cerr << "The number of Elements along z-axis must be larger than 1." );

   this -> dimensions.x() = xSize;
   this -> dimensions.y() = ySize;
   this -> dimensions.z() = zSize;
   dofs = zSize * ySize * xSize;
   
   VertexType parametricStep;
   parametricStep. x() = geometry. getProportions(). x() / xSize;
   parametricStep. y() = geometry. getProportions(). y() / ySize;
	parametricStep. z() = geometry. getProportions(). z() / zSize;
   geometry. setParametricStep( parametricStep );
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
bool tnlGrid< 3, Real, Device, Index, Geometry > :: setDimensions( const CoordinatesType& dimensions )
{
   return this -> setDimensions( dimensions. x(), dimensions. y(), dimensions. z() );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
const typename tnlGrid< 3, Real, Device, Index, Geometry > :: CoordinatesType& 
   tnlGrid< 3, Real, Device, Index, Geometry > :: getDimensions() const
{
   return this -> dimensions;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
void tnlGrid< 3, Real, Device, Index, Geometry > :: setOrigin( const VertexType& origin )
{
   this -> origin = origin;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
const typename tnlGrid< 3, Real, Device, Index, Geometry > :: VertexType&
   tnlGrid< 3, Real, Device, Index, Geometry > :: getOrigin() const
{
   return this -> origin;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
void tnlGrid< 3, Real, Device, Index, Geometry > :: setProportions( const VertexType& proportions )
{
   geometry. setProportions( proportions );
   VertexType parametricStep;
   parametricStep. x() = proportions. x() / ( this -> dimensions. x() );
   parametricStep. y() = proportions. y() / ( this -> dimensions. y() );
	parametricStep. z() = proportions. z() / ( this -> dimensions. z() );
   geometry. setParametricStep( parametricStep );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
const typename tnlGrid< 3, Real, Device, Index, Geometry > :: VertexType&
   tnlGrid< 3, Real, Device, Index, Geometry > :: getProportions() const
{
	return geometry. getProportions();
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
void tnlGrid< 3, Real, Device, Index, Geometry > :: setParametricStep( const VertexType& spaceStep )
{
      geometry. setProportions(
         VertexType(
            this -> dimensions. x() * geometry. getParametricStep(). x(),
            this -> dimensions. y() * geometry. getParametricStep(). y(),
				this -> dimensions. z() * geometry. getParametricStep(). z() ) );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
const typename tnlGrid< 3, Real, Device, Index, Geometry > :: VertexType&
   tnlGrid< 3, Real, Device, Index, Geometry > :: getParametricStep() const
{
   return geometry. getParametricStep();
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
Index tnlGrid< 3, Real, Device, Index, Geometry > :: getElementIndex( const Index i, const Index j, const Index k ) const
{
   tnlAssert( i < dimensions. x(),
              cerr << "Index i ( " << i
                   << " ) is out of range ( " << dimensions. x()
                   << " ) in tnlGrid " << this -> getName(); )
   tnlAssert( j < dimensions. y(),
              cerr << "Index j ( " << j
                   << " ) is out of range ( " << dimensions. y()
                   << " ) in tnlGrid " << this -> getName(); )
   tnlAssert( k < dimensions. z(),
            cerr << "Index k ( " << k
                 << " ) is out of range ( " << dimensions. z()
                 << " ) in tnlGrid " << this -> getName(); )

   return ( k * this -> dimensions. y() + j ) * this -> dimensions. x() + i;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
void tnlGrid< 3, Real, Device, Index, Geometry > :: getElementCoordinates( const Index element,
                                                                           CoordinatesType& coordinates ) const
{
   tnlAssert( element >= 0 && element < dofs,
              cerr << " element = " << element << " dofs = " << dofs
                   << " in tnlGrid " << this -> getName(); );

	coordinates. z() = element / (this -> dimensions. x()*this -> dimensions. y());
   coordinates. y() = (element - coordinates. z()*(this -> dimensions. x()*this -> dimensions. y())) / this -> dimensions. x();
	coordinates. x() = element - coordinates. z()*(this -> dimensions. x()*this -> dimensions. y()) - this -> dimensions. x()*coordinates. y();
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
void tnlGrid< 3, Real, Device, Index, Geometry > :: getElementCenter( const CoordinatesType& coordinates,
                                                                      VertexType& center ) const
{
      geometry. getElementCenter( origin, coordinates, center );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
Index tnlGrid< 3, Real, Device, Index, Geometry > :: getDofs() const
{
   return this -> dofs;
};

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
Real tnlGrid< 3, Real, Device, Index, Geometry > :: getElementMeasure( const CoordinatesType& coordinates ) const
{
   return geometry. getElementMeasure( coordinates );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
   template< int dx, int dy >
Real tnlGrid< 3, Real, Device, Index, Geometry > :: getDualElementMeasure( const CoordinatesType& coordinates ) const
{
   return 0.0; // TODO: fix this
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
   template< typename GridFunction >
      typename GridFunction::RealType
         tnlGrid< 3, Real, Device, Index, Geometry >::getDifferenceAbsMax( const GridFunction& f1,
                                                                           const GridFunction& f2 ) const
{
   typename GridFunction::RealType maxDiff( -1.0 );
   for( IndexType k = 0; k < getDimensions(). z(); k++ )
      for( IndexType j = 0; j < getDimensions(). y(); j++ )
         for( IndexType i = 0; i < getDimensions(). x(); i++ )
         {
            IndexType c = this -> getElementIndex( i, j, k );
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
         tnlGrid< 3, Real, Device, Index, Geometry >::getDifferenceLpNorm( const GridFunction& f1,
                                                                           const GridFunction& f2,
                                                                           const typename GridFunction::RealType& p ) const
{
   typename GridFunction::RealType lpNorm( 0.0 );
   for( IndexType k = 0; k < getDimensions(). z(); k++ )
      for( IndexType j = 0; j < getDimensions(). y(); j++ )
         for( IndexType i = 0; i < getDimensions(). x(); i++ )
         {
            IndexType c = this->getElementIndex( i, j, k );
            lpNorm += pow( tnlAbs( f1[ c ] - f2[ c ] ), p ) *
               this->getElementMeasure( CoordinatesType( i, j, k ) );
         }
   return pow( lpNorm, 1.0 / p );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
bool tnlGrid< 3, Real, Device, Index, Geometry > :: save( tnlFile& file ) const
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
bool tnlGrid< 3, Real, Device, Index, Geometry > :: load( tnlFile& file )
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
   return true;
};

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
bool tnlGrid< 3, Real, Device, Index, Geometry > :: save( const tnlString& fileName ) const
{
   return tnlObject :: save( fileName );
};

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
bool tnlGrid< 3, Real, Device, Index, Geometry > :: load( const tnlString& fileName )
{
   return tnlObject :: load( fileName );
};

template< typename Real,
           typename Device,
           typename Index,
           template< int, typename, typename, typename > class Geometry >
bool tnlGrid< 3, Real, Device, Index, Geometry >::writeMesh( const tnlString& fileName,
                                                             const tnlString& format ) const
{
   tnlAssert( false, cerr << "TODO: FIX THIS"); // TODO: FIX THIS
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
   template< typename MeshFunction >
bool tnlGrid< 3, Real, Device, Index, Geometry > :: write( const MeshFunction& function,
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
   if( format == "gnuplot" )
   {
      for( IndexType k = 0; k < getDimensions(). z(); k++ )
      {
         for( IndexType j = 0; j < getDimensions(). y(); j++ )
         {
            for( IndexType i = 0; i < getDimensions(). x(); i++ )
            {
               VertexType v;
               this -> getElementCenter( CoordinatesType( i, j, k ), v );
               tnlGnuplotWriter::write( file, v );
               tnlGnuplotWriter::write( file, function[ this -> getElementIndex( i, j, k ) ] );
               file << endl;
            }
         }
         file << endl;
      }
   }

   file. close();
   return true;
}

#endif /* TNLGRID3D_IMPL_H_ */
