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
          typename Index >
tnlGrid< 2, Real, Device, Index> :: tnlGrid()
: dofs( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlGrid< 2, Real, Device, Index> :: getTypeStatic()
{
   return tnlString( "tnlGrid< " ) +
          tnlString( Dimensions ) + ", " +
          tnlString( getParameterType< RealType >() ) + ", " +
          tnlString( Device :: getDeviceType() ) + ", " +
          tnlString( getParameterType< IndexType >() ) + " >";
}

template< typename Real,
           typename Device,
           typename Index >
tnlString tnlGrid< 2, Real, Device, Index> :: getType() const
{
   return this -> getTypeStatic();
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 2, Real, Device, Index> :: setDimensions( const Index ySize, const Index xSize )
{
   tnlAssert( xSize > 1,
              cerr << "The number of nodes along x-axis must be larger than 1." );
   tnlAssert( ySize > 1,
              cerr << "The number of nodes along y-axis must be larger than 1." );

   this -> dimensions. x() = xSize;
   this -> dimensions. y() = ySize;
   dofs = ySize * xSize;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 2, Real, Device, Index> :: setDimensions( const tnlTuple< 2, Index >& dimensions )
{
   tnlAssert( dimensions. x() > 1,
              cerr << "The number of nodes along x-axis must be larger than 1." );
   tnlAssert( dimensions. y() > 1,
              cerr << "The number of nodes along y-axis must be larger than 1." );

   this -> dimensions = dimensions;
   dofs = this -> dimensions. x() * this -> dimensions. y();
}

template< typename Real,
          typename Device,
          typename Index >
const tnlTuple< 2, Index >& tnlGrid< 2, Real, Device, Index> :: getDimensions() const
{
   return this -> dimensions;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 2, Real, Device, Index> :: setLowerCorner( const tnlTuple< 2, Real >& lowerCorner )
{
   this -> lowerCorner = lowerCorner;
}

template< typename Real,
          typename Device,
          typename Index >
const tnlTuple< 2, Real >& tnlGrid< 2, Real, Device, Index> :: getLowerCorner() const
{
   return this -> lowerCorner;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 2, Real, Device, Index> :: setUpperCorner( const tnlTuple< 2, Real >& upperCorner )
{
   this -> upperCorner = upperCorner;
}

template< typename Real,
          typename Device,
          typename Index >
const tnlTuple< 2, Real >& tnlGrid< 2, Real, Device, Index> :: getUpperCorner() const
{
   return this -> upperCorner;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 2, Real, Device, Index> :: setSpaceStep( const tnlTuple< 2, Real >& spaceStep )
{
   this -> upperCorner. x() = this -> lowerCorner. x() +
                              this -> dimensions. x() *
                              spaceStep. x();
   this -> upperCorner. y() = this -> lowerCorner. y() +
                              this -> dimensions. y() *
                              spaceStep. y();

}

template< typename Real,
          typename Device,
          typename Index >
tnlTuple< 2, Real > tnlGrid< 2, Real, Device, Index> :: getSpaceStep() const
{
   tnlAssert( dimensions. x() > 0,
              cerr << "Cannot get the space step hx since number of nodes along the x axis is not known in tnlGrid "
                   << this -> getName() );
   tnlAssert( dimensions. y() > 0,
              cerr << "Cannot get the space step hy since number of nodes along the y axis is not known in tnlGrid "
                   << this -> getName() );

   tnlTuple< 2, RealType > spaceStep;
   spaceStep. x() =
            ( this -> upperCorner. x() - this -> lowerCorner. x() ) /
            ( Real ) ( this -> dimensions. x() - 1 );
   spaceStep. y() =
            ( this -> upperCorner. y() - this -> lowerCorner. y() ) /
            ( Real ) ( this -> dimensions. y() - 1 );
   return spaceStep;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlGrid< 2, Real, Device, Index> :: getNodeIndex( const Index j, const Index i ) const
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
          typename Index >
Index tnlGrid< 2, Real, Device, Index> :: getNodeNeighbour( const Index node,
                                                            const Index dy,
                                                            const Index dx ) const
{
   tnlAssert( node + dy * this -> dimensions. x() + dx < getDofs(),
              cerr << "Index of neighbour with dx = " << dx
                   << " and dy = " << dy
                   << " is out of range ( " << dimensions. x()
                   << " ) in tnlGrid " << this -> getName(); )
   return node + dy * this -> dimensions. x() + dx;
}


template< typename Real,
           typename Device,
           typename Index >
Index tnlGrid< 2, Real, Device, Index> :: getDofs() const
{
   return this -> dofs;
};

template< typename Real,
          typename Device,
          typename Index >
bool tnlGrid< 2, Real, Device, Index> :: save( tnlFile& file ) const
{
   if( ! tnlObject :: save( file ) )
      return false;
   if( ! this -> lowerCorner. save( file ) ||
       ! this -> upperCorner. save( file ) ||
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
          typename Index >
bool tnlGrid< 2, Real, Device, Index> :: load( tnlFile& file )
{
   if( ! tnlObject :: load( file ) )
      return false;
   if( ! this -> lowerCorner. load( file ) ||
       ! this -> upperCorner. load( file ) ||
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
          typename Index >
bool tnlGrid< 2, Real, Device, Index> :: save( const tnlString& fileName ) const
{
   return tnlObject :: save( fileName );
};

template< typename Real,
           typename Device,
           typename Index >
bool tnlGrid< 2, Real, Device, Index> :: load( const tnlString& fileName )
{
   return tnlObject :: load( fileName );
};

template< typename Real,
           typename Device,
           typename Index >
   template< typename MeshFunction >
bool tnlGrid< 2, Real, Device, Index> :: write( const MeshFunction& function,
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
            const RealType x = this -> getLowerCorner(). x() + i * hx;
            const RealType y = this -> getLowerCorner(). y() + j * hy;
            file << x << " " << " " << y << " " << function[ this -> getNodeIndex( j, i ) ] << endl;
         }
         file << endl;
      }

   file. close();
   return true;
}


#endif /* TNLGRID2D_IMPL_H_ */
