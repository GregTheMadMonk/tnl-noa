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
#include <core/tnlString.h>
#include <core/tnlAssert.h>

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
tnlString tnlGrid< 1, Real, Device, Index, Geometry > :: getTypeStatic()
{
   return tnlString( "tnlGrid< " ) +
          tnlString( Dimensions ) + ", " +
          tnlString( getParameterType< RealType >() ) + ", " +
          tnlString( Device :: getDeviceType() ) + ", " +
          tnlString( getParameterType< IndexType >() ) + " >";
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
tnlString tnlGrid< 1, Real, Device, Index, Geometry > :: getType() const
{
   return this -> getTypeStatic();
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry  >
void tnlGrid< 1, Real, Device, Index, Geometry > :: setDimensions( const Index xSize )
{
   tnlAssert( xSize > 0,
              cerr << "The number of Elements along x-axis must be larger than 0." );
   this -> dimensions. x() = xSize;
   dofs = xSize;

   VertexType parametricStep;
   parametricStep. x() = proportions. x() / xSize;
   geometry. setParametricStep( parametricStep );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry  >
void tnlGrid< 1, Real, Device, Index, Geometry > :: setDimensions( const CoordinatesType& dimensions )
{
   this -> setDimensions( dimensions. x() );
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
   this -> proportions = proportions;
   this -> setDimensions( this -> dimensions );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry  >
const typename tnlGrid< 1, Real, Device, Index, Geometry > :: VertexType& 
   tnlGrid< 1, Real, Device, Index, Geometry > :: getProportions() const
{
   return this -> proportions;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry  >
void tnlGrid< 1, Real, Device, Index, Geometry > :: setParametricStep( const VertexType& parametricStep )
{
   this -> proportions. x() = this -> dimensions. x() * parametricStep. x();
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
Index tnlGrid< 1, Real, Device, Index, Geometry > :: getDofs() const
{
   return this -> dofs;
};

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
bool tnlGrid< 1, Real, Device, Index, Geometry > :: save( tnlFile& file ) const
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
bool tnlGrid< 1, Real, Device, Index, Geometry > :: load( tnlFile& file )
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
   this -> dofs = this -> getDimensions(). x();
   return true;
};

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
bool tnlGrid< 1, Real, Device, Index, Geometry > :: save( const tnlString& fileName ) const
{
   return tnlObject :: save( fileName );
};

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
bool tnlGrid< 1, Real, Device, Index, Geometry > :: load( const tnlString& fileName )
{
   return tnlObject :: load( fileName );
};

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
   const RealType hx = getParametricStep(). x();
   const RealType hy = getParametricStep(). y();
   if( format == "gnuplot" )
      for( IndexType i = 0; i < getDimensions(). x(); i++ )
      {
         const RealType x = this -> getOrigin(). x() + i * hx;
         file << x << " " << function[ this -> getElementIndex( i ) ] << endl;
      }

   file. close();
   return true;
}

#endif /* TNLGRID1D_IMPL_H_ */
