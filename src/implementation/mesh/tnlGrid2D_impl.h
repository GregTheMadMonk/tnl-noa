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
void tnlGrid< 2, Real, Device, Index, Geometry > :: setDimensions( const Index xSize, const Index ySize )
{
   tnlAssert( xSize > 0,
              cerr << "The number of Elements along x-axis must be larger than 0." );
   tnlAssert( ySize > 0,
              cerr << "The number of Elements along y-axis must be larger than 0." );

   this -> dimensions. x() = xSize;
   this -> dimensions. y() = ySize;
   dofs = ySize * xSize;
   tnlTuple< 2, Real > parametricStep;
   parametricStep. x() = proportions. x() / xSize;
   parametricStep. y() = proportions. y() / ySize;
   geometry. setParametricStep( parametricStep );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
void tnlGrid< 2, Real, Device, Index, Geometry > :: setDimensions( const tnlTuple< 2, Index >& dimensions )
{
   this -> setDimensions( dimensions. x(), dimensions. y() );
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
   tnlTuple< 2, Real > parametricStep;
   parametricStep. x() = proportions. x() / ( this -> dimensions. x() - 1 );
   parametricStep. y() = proportions. y() / ( this -> dimensions. y() - 1 );
   geometry. setParametricStep( parametricStep );
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
void tnlGrid< 2, Real, Device, Index, Geometry > :: setParametricStep( const tnlTuple< 2, Real >& spaceStep )
{
   this -> proportions. x() = this -> dimensions. x() *
                              geometry. getParametricStep(). x();
   this -> proportions. y() = this -> dimensions. y() *
                              geometry. getParametricStep(). y();
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
const tnlTuple< 2, Real >& tnlGrid< 2, Real, Device, Index, Geometry > :: getParametricStep() const
{
   /*tnlAssert( dimensions. x() > 0,
              cerr << "Cannot get the space step hx since number of Elements along the x axis is not known in tnlGrid "
                   << this -> getName() );
   tnlAssert( dimensions. y() > 0,
              cerr << "Cannot get the space step hy since number of Elements along the y axis is not known in tnlGrid "
                   << this -> getName() );

   tnlTuple< 2, RealType > parametricStep;
   parametricStep. x() =
            this -> proportions. x() / ( Real ) ( this -> dimensions. x() - 1 );
   parametricStep. y() =
            this -> proportions. y() / ( Real ) ( this -> dimensions. y() - 1 );*/
   return geometry. getParametricStep();
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
Index tnlGrid< 2, Real, Device, Index, Geometry > :: getElementIndex( const Index i, const Index j ) const
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
void tnlGrid< 2, Real, Device, Index, Geometry > :: getElementCoordinates( const Index element,
                                                                           tnlTuple< 2, Index >& coordinates ) const
{
   tnlAssert( i >= 0 && i < dofs,
              cerr << " i = " << i << " dofs = " << dofs
                   << " in tnlGrid " << this -> getName(); );

   coordinates. x() = element % this -> dimensions. x();
   coordinates. y() = element / this -> dimensions. x();
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
void tnlGrid< 2, Real, Device, Index, Geometry > :: getElementCenter( const tnlTuple< 2, Index >& coordinates,
                                                                      tnlTuple< 2, Real >& center ) const
{
      geometry. getElementCenter( origin, coordinates, center );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
Real tnlGrid< 2, Real, Device, Index, Geometry > :: getElementMeasure( const tnlTuple< 2, Index >& coordinates ) const
{
   return geometry. getElementMeasure( coordinates );
}

/*template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
   template< Index dy, Index dx >
Real tnlGrid< 2, Real, Device, Index, Geometry > :: getElementsDistance( const Index j,
                                                                         const Index i ) const
{
   return geometry. getElementsDistance< dy, dx >( j, i );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
template< int dy, int dx >
Real tnlGrid< 2, Real, Device, Index, Geometry > :: getEdgeLength( const Index j,
                                                                   const Index i ) const
{
   return geometry. getEdgeLength< dy, dx >( j, i );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
template< int dy, int dx >
tnlTuple< 2, Real > tnlGrid< 2, Real, Device, Index, Geometry > :: getEdgeNormal( const Index j,
                                                                                  const Index i ) const
{
   return geometry. getEdgeNormal< dy, dx >( j, i );
}*/

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
   if( ! geometry. save( file ) )
      return false;
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
   if( ! geometry. load( file ) )
      return false;
   this -> dofs = this -> getDimensions(). x() *
                  this -> getDimensions(). y();
   tnlTuple< 2, Real > parametricStep;
   parametricStep. x() = proportions. x() / ( this -> dimensions. x() - 1 );
   parametricStep. y() = proportions. y() / ( this -> dimensions. y() - 1 );
   geometry. setParametricStep( parametricStep );
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
   const RealType hx = getParametricStep(). x();
   const RealType hy = getParametricStep(). y();
   if( format == "gnuplot" )
      for( IndexType j = 0; j < getDimensions(). y(); j++ )
      {
         for( IndexType i = 0; i < getDimensions(). x(); i++ )
         {
            const RealType x = this -> getOrigin(). x() + i * hx;
            const RealType y = this -> getOrigin(). y() + j * hy;
            file << x << " " << " " << y << " " << function[ this -> getElementIndex( i, j ) ] << endl;
         }
         file << endl;
      }

   file. close();
   return true;
}


#endif /* TNLGRID2D_IMPL_H_ */
