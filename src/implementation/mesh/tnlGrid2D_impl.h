/***************************************************************************
                          tnlGrid2D_impl.h  -  description
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

#ifndef TNLGRID2D_IMPL_H_
#define TNLGRID2D_IMPL_H_

#include <fstream>
#include <core/tnlAssert.h>

using namespace std;

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry  >
tnlGrid< 2, Real, Device, Index, Geometry > :: tnlGrid()
: dofs( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry  >
tnlString tnlGrid< 2, Real, Device, Index, Geometry > :: getTypeStatic()
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
tnlString tnlGrid< 2, Real, Device, Index, Geometry > :: getType() const
{
   return this -> getTypeStatic();
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
void tnlGrid< 2, Real, Device, Index, Geometry > :: setDimensions( const Index ySize, const Index xSize )
{
   tnlAssert( xSize > 1,
              cerr << "The number of Elements along x-axis must be larger than 1." );
   tnlAssert( ySize > 1,
              cerr << "The number of Elements along y-axis must be larger than 1." );

   this -> dimensions. x() = xSize;
   this -> dimensions. y() = ySize;
   dofs = ySize * xSize;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
void tnlGrid< 2, Real, Device, Index, Geometry > :: setDimensions( const tnlTuple< 2, Index >& dimensions )
{
   tnlAssert( dimensions. x() > 1,
              cerr << "The number of Elements along x-axis must be larger than 1." );
   tnlAssert( dimensions. y() > 1,
              cerr << "The number of Elements along y-axis must be larger than 1." );

   this -> dimensions = dimensions;
   dofs = this -> dimensions. x() * this -> dimensions. y();
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
const tnlTuple< 2, Index >& tnlGrid< 2, Real, Device, Index, Geometry > :: getDimensions() const
{
   return this -> dimensions;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
void tnlGrid< 2, Real, Device, Index, Geometry > :: setOrigin( const tnlTuple< 2, Real >& origin )
{
   this -> origin = origin;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
const tnlTuple< 2, Real >& tnlGrid< 2, Real, Device, Index, Geometry > :: getOrigin() const
{
   return this -> origin;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
void tnlGrid< 2, Real, Device, Index, Geometry > :: setProportions( const tnlTuple< 2, Real >& proportions )
{
   this -> proportions = proportions;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
const tnlTuple< 2, Real >& tnlGrid< 2, Real, Device, Index, Geometry > :: getProportions() const
{
   return this -> proportions;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
void tnlGrid< 2, Real, Device, Index, Geometry > :: setSpaceStep( const tnlTuple< 2, Real >& spaceStep )
{
   this -> proportions. x() = this -> dimensions. x() *
                              spaceStep. x();
   this -> proportions. y() = this -> dimensions. y() *
                              spaceStep. y();

}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
tnlTuple< 2, Real > tnlGrid< 2, Real, Device, Index, Geometry > :: getSpaceStep() const
{
   tnlAssert( dimensions. x() > 0,
              cerr << "Cannot get the space step hx since number of Elements along the x axis is not known in tnlGrid "
                   << this -> getName() );
   tnlAssert( dimensions. y() > 0,
              cerr << "Cannot get the space step hy since number of Elements along the y axis is not known in tnlGrid "
                   << this -> getName() );

   tnlTuple< 2, RealType > spaceStep;
   spaceStep. x() =
            this -> proportions. x() / ( Real ) ( this -> dimensions. x() - 1 );
   spaceStep. y() =
            this -> proportions. y() / ( Real ) ( this -> dimensions. y() - 1 );
   return spaceStep;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
Index tnlGrid< 2, Real, Device, Index, Geometry > :: getElementIndex( const Index j, const Index i ) const
{
   tnlAssert( i < dimensions. x(),
              cerr << "Index i ( " << i
                   << " ) is out of range ( " << dimensions. x()
                   << " ) in tnlGrid " << this -> getName(); )
   tnlAssert( j < dimensions. y(),
              cerr << "Index j ( " << j
                   << " ) is out of range ( " << dimensions. y()
                   << " ) in tnlGrid " << this -> getName(); )

   return j * this -> dimensions. x() + i;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
Index tnlGrid< 2, Real, Device, Index, Geometry > :: getElementNeighbour( const Index Element,
                                                            const Index dy,
                                                            const Index dx ) const
{
   tnlAssert( Element + dy * this -> dimensions. x() + dx < getDofs(),
              cerr << "Index of neighbour with dx = " << dx
                   << " and dy = " << dy
                   << " is out of range ( " << dimensions. x()
                   << " ) in tnlGrid " << this -> getName(); )
   return Element + dy * this -> dimensions. x() + dx;
}


template< typename Real,
           typename Device,
           typename Index,
           template< int, typename, typename, typename > class Geometry >
Index tnlGrid< 2, Real, Device, Index, Geometry > :: getDofs() const
{
   return this -> dofs;
};

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
bool tnlGrid< 2, Real, Device, Index, Geometry > :: save( tnlFile& file ) const
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
bool tnlGrid< 2, Real, Device, Index, Geometry > :: load( tnlFile& file )
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
                   this -> getDimensions(). y();
   return true;
};

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
bool tnlGrid< 2, Real, Device, Index, Geometry > :: save( const tnlString& fileName ) const
{
   return tnlObject :: save( fileName );
};

template< typename Real,
           typename Device,
           typename Index,
           template< int, typename, typename, typename > class Geometry >
bool tnlGrid< 2, Real, Device, Index, Geometry > :: load( const tnlString& fileName )
{
   return tnlObject :: load( fileName );
};

template< typename Real,
           typename Device,
           typename Index,
           template< int, typename, typename, typename > class Geometry >
   template< typename MeshFunction >
bool tnlGrid< 2, Real, Device, Index, Geometry > :: write( const MeshFunction& function,
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
   const RealType hx = getSpaceStep(). x();
   const RealType hy = getSpaceStep(). y();
   if( format == "gnuplot" )
      for( IndexType j = 0; j < getDimensions(). y(); j++ )
      {
         for( IndexType i = 0; i < getDimensions(). x(); i++ )
         {
            const RealType x = this -> getOrigin(). x() + i * hx;
            const RealType y = this -> getOrigin(). y() + j * hy;
            file << x << " " << " " << y << " " << function[ this -> getElementIndex( j, i ) ] << endl;
         }
         file << endl;
      }

   file. close();
   return true;
}


#endif /* TNLGRID2D_IMPL_H_ */
