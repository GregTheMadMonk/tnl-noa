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
void tnlGrid< 3, Real, Device, Index, Geometry > :: setDimensions( const Index xSize, const Index ySize, const Index zSize )
{
   tnlAssert( xSize > 1,
              cerr << "The number of Elements along x-axis must be larger than 1." );
   tnlAssert( ySize > 1,
              cerr << "The number of Elements along y-axis must be larger than 1." );
   tnlAssert( zSize > 1,
              cerr << "The number of Elements along z-axis must be larger than 1." );

   this -> dimensions. x() = xSize;
   this -> dimensions. y() = ySize;
   this -> dimensions. z() = zSize;
   dofs = zSize * ySize * xSize;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
void tnlGrid< 3, Real, Device, Index, Geometry > :: setDimensions( const CoordinatesType& dimensions )
{
   this -> setDimensions( this -> dimensions. x(), this -> dimensions. y(), this -> dimensions. z() );
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
   this -> proportions = proportions;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
const typename tnlGrid< 3, Real, Device, Index, Geometry > :: VertexType&
   tnlGrid< 3, Real, Device, Index, Geometry > :: getProportions() const
{
   return this -> proportions;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
void tnlGrid< 3, Real, Device, Index, Geometry > :: setParametricStep( const VertexType& spaceStep )
{
   this -> proportions. x() = this -> dimensions. x() *
                              spaceStep. x();
   this -> proportions. y() = this -> dimensions. y() *
                              spaceStep. y();
   this -> proportions. z() = this -> dimensions. z() *
                              spaceStep. z();
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
const typename tnlGrid< 3, Real, Device, Index, Geometry > :: VertexType&
   tnlGrid< 3, Real, Device, Index, Geometry > :: getParametricStep() const
{
   //return geometry. getParametricStep();
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
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
void tnlGrid< 3, Real, Device, Index, Geometry > :: getElementCenter( const CoordinatesType& coordinates,
                                                                      VertexType& center ) const
{
      //geometry. getElementCenter( origin, coordinates, center );
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
   return 0.0; // TODO: fix this
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
            lpNorm += pow( p, tnlAbs( f1[ c ] - f2[ c ] ) ) *
               this->getElementMeasure( CoordinatesType( i, j, k ) );
         }
   return pow( 1.0 / p, lpNorm );
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
       ! this -> proportions. save( file ) ||
       ! this -> dimensions. save( file ) )
   {
      cerr << "I was not able to save the domain description of the tnlGrid "
           << this -> getName() << endl;
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
   if( ! this -> origin. load( file ) ||
       ! this -> proportions. load( file ) ||
       ! this -> dimensions. load( file ) )
   {
      cerr << "I was not able to load the domain description of the tnlGrid "
           << this -> getName() << endl;
      return false;
   }
   this -> dofs = this -> getDimensions(). x() *
                  this -> getDimensions(). y() *
                  this -> getDimensions(). z();
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
   const RealType hx = getParametricStep(). x();
   const RealType hy = getParametricStep(). y();
   if( format == "gnuplot" )
   {
      tnlAssert( false, cerr << "TODO");
      for( IndexType j = 0; j < getDimensions(). y(); j++ )
      {
         for( IndexType i = 0; i < getDimensions(). x(); i++ )
         {
            const RealType x = this -> getOrigin(). x() + i * hx;
            const RealType y = this -> getOrigin(). y() + j * hy;
            //file << x << " " << " " << y << " " << function[ this -> getElementIndex( i, j ) ] << endl;
         }
         file << endl;
      }
   }

   file. close();
   return true;
}

#endif /* TNLGRID3D_IMPL_H_ */
