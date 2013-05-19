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
   VertexType parametricStep;
   parametricStep. x() = proportions. x() / xSize;
   parametricStep. y() = proportions. y() / ySize;
   geometry. setParametricStep( parametricStep );
   if( GeometryType :: ElementMeasureStorage :: enabled )
   {
      elementsMeasure. setSize( this -> getDofs() );
      dualElementsMeasure. setSize( this -> getDofs() );
      edgeNormals. setSize( this -> getNumberOfEdges() );
   }
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
void tnlGrid< 2, Real, Device, Index, Geometry > :: setDimensions( const CoordinatesType& dimensions )
{
   this -> setDimensions( dimensions. x(), dimensions. y() );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
const typename tnlGrid< 2, Real, Device, Index, Geometry > :: CoordinatesType& 
   tnlGrid< 2, Real, Device, Index, Geometry > :: getDimensions() const
{
   return this -> dimensions;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
void tnlGrid< 2, Real, Device, Index, Geometry > :: setOrigin( const VertexType& origin )
{
   this -> origin = origin;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
const typename tnlGrid< 2, Real, Device, Index, Geometry > :: VertexType& 
   tnlGrid< 2, Real, Device, Index, Geometry > :: getOrigin() const
{
   return this -> origin;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
void tnlGrid< 2, Real, Device, Index, Geometry > :: setProportions( const VertexType& proportions )
{
   this -> proportions = proportions;
   VertexType parametricStep;
   parametricStep. x() = proportions. x() / ( this -> dimensions. x() - 1 );
   parametricStep. y() = proportions. y() / ( this -> dimensions. y() - 1 );
   geometry. setParametricStep( parametricStep );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
const typename tnlGrid< 2, Real, Device, Index, Geometry > :: VertexType& 
   tnlGrid< 2, Real, Device, Index, Geometry > :: getProportions() const
{
   return this -> proportions;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
void tnlGrid< 2, Real, Device, Index, Geometry > :: setParametricStep( const VertexType& spaceStep )
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
const typename tnlGrid< 2, Real, Device, Index, Geometry > :: VertexType& 
   tnlGrid< 2, Real, Device, Index, Geometry > :: getParametricStep() const
{
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
                   << " ) in tnlGrid " << this -> getName(); );
   tnlAssert( j < dimensions. y(),
              cerr << "Index j ( " << j
                   << " ) is out of range ( " << dimensions. y()
                   << " ) in tnlGrid " << this -> getName(); );

   return j * this -> dimensions. x() + i;
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
Index tnlGrid< 2, Real, Device, Index, Geometry > :: getEdgeIndex( const Index i,
                                                                   const Index j,
                                                                   const Index dx,
                                                                   const Index dy ) const
{
   tnlAssert( i < dimensions. x(),
              cerr << "Index i ( " << i
                   << " ) is out of range ( " << dimensions. x()
                   << " ) in tnlGrid " << this -> getName(); );
   tnlAssert( j < dimensions. y(),
              cerr << "Index j ( " << j
                   << " ) is out of range ( " << dimensions. y()
                   << " ) in tnlGrid " << this -> getName(); );
   tnlAssert( dx == 0 && ( dy == 1 || dy == -1 ) ||
              dy == 0 && ( dx == 1 || dx == -1 ),
              cerr << "dx = " << dx << ", dy = " << dy << endl;);
   return ( j + dy ) * this -> dimensions. x() + i + dx;
}


template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
void tnlGrid< 2, Real, Device, Index, Geometry > :: refresh()
{
   if( GeometryType :: ElementMeasureStorage :: enabled )
      for( IndexType j = 0; j < dimensions. y(); j ++ )
         for( IndexType i = 0; i < dimensions. x(); i ++ )
            elementsMeasure[ getElementIndex( i, j ) ] =
                     geometry. getElementMeasure( CoordinatesType( i, j ) );

   if( GeometryType :: DualElementMeasureStorage :: enabled )
      for( IndexType j = 1; j < dimensions. y() - 1; j ++ )
         for( IndexType i = 1; i < dimensions. x() - 1; i ++ )
         {
            dualElementsMeasure[ getElementIndex( i + 1, j ) ] =
                     geometry. getDualElementMeasure<  1,  0 >( CoordinatesType( i, j ) );
            dualElementsMeasure[ getElementIndex( i - 1, j ) ] =
                     geometry. getDualElementMeasure< -1,  0 >( CoordinatesType( i, j ) );
            dualElementsMeasure[ getElementIndex( i, j + 1 ) ] =
                     geometry. getDualElementMeasure<  0,  1 >( CoordinatesType( i, j ) );
            dualElementsMeasure[ getElementIndex( i, j - 1 ) ] =
                     geometry. getDualElementMeasure<  0, -1 >( CoordinatesType( i, j ) );
         }

   if( GeometryType :: EdgeNormalStorage :: enabled )
      for( IndexType j = 0; j < dimensions. y(); j ++ )
         for( IndexType i = 0; i < dimensions. x(); i ++ )
         {
            geometry. getEdgeNormal<  1,  0 >( CoordinatesType( i, j ),
                                               edgeNormals[ getEdgeIndex( i, j, 1, 0 ) ] );
            geometry. getEdgeNormal< -1,  0 >( CoordinatesType( i, j ),
                                                edgeNormals[ getEdgeIndex( i, j, -1, 0 ) ] );
            geometry. getEdgeNormal<  0,  1 >( CoordinatesType( i, j ),
                                               edgeNormals[ getEdgeIndex( i, j, 0, 1 ) ]  );
            geometry. getEdgeNormal<  0, -1 >( CoordinatesType( i, j ),
                                               edgeNormals[ getEdgeIndex( i, j, 0, -1 ) ] );
         }
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
void tnlGrid< 2, Real, Device, Index, Geometry > :: getElementCoordinates( const Index element,
                                                                           CoordinatesType& coordinates ) const
{
   tnlAssert( element >= 0 && element < dofs,
              cerr << " element = " << element << " dofs = " << dofs
                   << " in tnlGrid " << this -> getName(); );

   coordinates. x() = element % this -> dimensions. x();
   coordinates. y() = element / this -> dimensions. x();
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
Index tnlGrid< 2, Real, Device, Index, Geometry > :: getElementNeighbour( const Index Element,
                                                                          const Index dx,
                                                                          const Index dy ) const
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
Index tnlGrid< 2, Real, Device, Index, Geometry > :: getNumberOfEdges() const
{
   return ( this -> dimensions. x() + 1 ) * ( this -> dimensions. y() + 1 );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
void tnlGrid< 2, Real, Device, Index, Geometry > :: getElementCenter( const CoordinatesType& coordinates,
                                                                      VertexType& center ) const
{
      geometry. getElementCenter( origin, coordinates, center );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
Real tnlGrid< 2, Real, Device, Index, Geometry > :: getElementMeasure( const CoordinatesType& coordinates ) const
{
   if( GeometryType :: ElementMeasureStorage :: enabled )
      return elementsMeasure[ getElementIndex( coordinates. x(), coordinates. y() ) ];
   return geometry. getElementMeasure( coordinates );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
   template< int dx, int dy >
Real tnlGrid< 2, Real, Device, Index, Geometry > :: getDualElementMeasure( const CoordinatesType& coordinates ) const
{
   if( GeometryType :: DualElementMeasureStorage :: enabled )
      return dualElementsMeasure[ getElementIndex( coordinates. x() + dx, coordinates. y() + dy ) ];
   return geometry. getDualElementMeasure< dx, dy >( coordinates );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
template< int dx, int dy >
void tnlGrid< 2, Real, Device, Index, Geometry > :: getEdgeNormal( const CoordinatesType& coordinates,
                                                                   VertexType& normal ) const
{
   //if( GeometryType :: EdgeNormalStorage :: enabled )
   //   normal = edgeNormals[ getEdgeIndex( coordinates. x(), coordinates. y(), dx, dy ) ];
   //else
      return geometry. getEdgeNormal< dx, dy >( coordinates, normal );
}

template< typename Real,
          typename Device,
          typename Index,
          template< int, typename, typename, typename > class Geometry >
   template< int dx, int dy >
void tnlGrid< 2, Real, Device, Index, Geometry > :: getVertex( const CoordinatesType& elementCoordinates,
                                                               VertexType& vertex ) const
{
   return geometry. getVertex< dx, dy >( elementCoordinates,
                                         this -> origin,
                                         vertex );
}

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
   VertexType parametricStep;
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
   if( format == "gnuplot" )
      for( IndexType j = 0; j < getDimensions(). y(); j++ )
      {
         for( IndexType i = 0; i < getDimensions(). x(); i++ )
         {
            VertexType v;
            this -> getElementCenter( CoordinatesType( i, j ), v );
            file << v. x() << " " << " " << v. y() << " " << function[ this -> getElementIndex( i, j ) ] << endl;
         }
         file << endl;
      }

   file. close();
   return true;
}


#endif /* TNLGRID2D_IMPL_H_ */
