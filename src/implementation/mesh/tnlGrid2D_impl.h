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
#include <iomanip>
#include <core/tnlAssert.h>
#include <mesh/tnlGnuplotWriter.h>

using namespace std;

template< typename Real,
          typename Device,
          typename Index >
tnlGrid< 2, Real, Device, Index > :: tnlGrid()
: dofs( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlGrid< 2, Real, Device, Index > :: getType()
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
tnlString tnlGrid< 2, Real, Device, Index > :: getTypeVirtual() const
{
   return this -> getType();
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 2, Real, Device, Index > :: setDimensions( const Index xSize, const Index ySize )
{
   tnlAssert( xSize > 0,
              cerr << "xSize = " << xSize );
   tnlAssert( ySize > 0,
              cerr << "ySize = " << ySize );

   this->dimensions.x() = xSize;
   this->dimensions.y() = ySize;
   this->numberOfCells = xSize * ySize;
   this->numberOfNxFaces = ySize * ( xSize + 1 );
   this->numberOfFaces = this->numberOfNxFaces + xSize * ( ySize + 1 );
   this->numberOfVertices = ( xSize + 1 ) * ( ySize + 1 );
   this->cellProportions.x() = this->proportions  / ( Real ) xSize;
   this->cellProportions.y() = this->proportions  / ( Real ) ySize;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlGrid< 2, Real, Device, Index > :: setDimensions( const CoordinatesType& dimensions )
{
   return this->setDimensions( dimensions. x(), dimensions. y() );
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
const typename tnlGrid< 2, Real, Device, Index >::CoordinatesType&
tnlGrid< 2, Real, Device, Index > :: getDimensions() const
{
   return this->dimensions;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 2, Real, Device, Index > :: setDomain( const VertexType& origin,
                                                     const VertexType& proportions )
{
   this->origin = origin;
   this->proportions = proportions;
   if( this->getDimensions().x() > 0 && this->getDimensions().y() > 0 )
   {
      this->cellProportions.x() = proportions.x() / ( Real ) this->getDimensions().x();
      this->cellProportions.y() = proportions.y() / ( Real ) this->getDimensions().y();
   }
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
const typename tnlGrid< 2, Real, Device, Index >::VertexType&
tnlGrid< 2, Real, Device, Index >::getOrigin() const
{
   return this->origin;
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
const typename tnlGrid< 2, Real, Device, Index > :: VertexType&
   tnlGrid< 2, Real, Device, Index > :: getProportions() const
{
   return this->proportions;
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
const typename tnlGrid< 2, Real, Device, Index > :: VertexType&
tnlGrid< 2, Real, Device, Index > :: getCellProportions() const
{
   return this->cellProportions;
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Index tnlGrid< 2, Real, Device, Index > :: getCellIndex( const CoordinatesType& cellCoordinates ) const
{
   tnlAssert( cellCoordinates.x() >= 0 && cellCoordinates.x() < this->getDimensions.x(),
              cerr << "cellCoordinates.x() = " << cellCoordinates.x()
                   << " this->getDimensions().x() = " << this->getDimensions().x()
                   << " this->getName() = " << this->getName(); );
   tnlAssert( cellCoordinates.y() >= 0 && cellCoordinates.y() < this->getDimensions.y(),
              cerr << "cellCoordinates.y() = " << cellCoordinates.y()
                   << " this->getDimensions().y() = " << this->getDimensions().y()
                   << " this->getName() = " << this->getName(); )

   return cellCoordinates.y() * this->dimensions.x() + cellCoordinates.x();
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
typename tnlGrid< 2, Real, Device, Index >::CoordinatesType
tnlGrid< 2, Real, Device, Index >::getCellCoordinates( const Index cellIndex ) const
{
   tnlAssert( cellIndex >= 0 && cellIndex < this->getNumberOfCells(),
              cerr << " cellIndex = " << cellIndex
                   << " this->getNumberOfCells() = " << this->getNumberOfCells()
                   << " this->getName() " << this->getName(); );
   return CoordinatesType( cellIndex % this->getDimensions.x(), cellIndex / this->getDimensions.x() );
}

template< typename Real,
          typename Device,
          typename Index >
template< int nx, int ny >
#ifdef HAVE_CUDA
__device__ __host__
#endif
Index tnlGrid< 2, Real, Device, Index >::getFaceIndex( const CoordinatesType& faceCoordinates ) const
{
   tnlStaticAssert( nx >= 0 && ny >= 0 && nx + ny = 1,);
   if( nx )
   {
      tnlAssert( faceCoordinates.x() >= 0 && faceCoordinates.x() < this->getDimensions.x() + 1,
                 cerr << "faceCoordinates.x() = " << faceCoordinates.x()
                      << " this->getDimensions().x() + 1 = " << this->getDimensions().x() + 1
                      << " this->getName() = " << this->getName(); );
      tnlAssert( faceCoordinates.y() >= 0 && faceCoordinates.y() < this->getDimensions.y(),
                 cerr << "faceCoordinates.y() = " << faceCoordinates.y()
                      << " this->getDimensions().y() = " << this->getDimensions().y()
                      << " this->getName() = " << this->getName(); )
      return faceCoordinates.y() * ( this->getDimensions().x() + 1 ) + faceCoordinates.x();
   }
   if( ny )
   {
      tnlAssert( faceCoordinates.x() >= 0 && faceCoordinates.x() < this->getDimensions.x(),
                 cerr << "faceCoordinates.x() = " << faceCoordinates.x()
                      << " this->getDimensions().x() = " << this->getDimensions().x()
                      << " this->getName() = " << this->getName(); );
      tnlAssert( faceCoordinates.y() >= 0 && faceCoordinates.y() < this->getDimensions.y() + 1,
                 cerr << "faceCoordinates.y() = " << faceCoordinates.y()
                      << " this->getDimensions().y() + 1 = " << this->getDimensions().y() + 1
                      << " this->getName() = " << this->getName(); )
      return this->numberOfNxFaces + faceCoordinates.y() * this->getDimensions().x() + faceCoordinates.x();
   }
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
__device__ __host__
#endif
typename tnlGrid< 2, Real, Device, Index >::CoordinatesType
tnlGrid< 2, Real, Device, Index >::getFaceCoordinates( const Index faceIndex, int& nx, int& ny ) const
{
   tnlAssert( faceIndex >= 0 && faceIndex < this->getNumberOfFaces(),
              cerr << " faceIndex = " << faceIndex
                   << " this->getNumberOfFaces() = " << this->getNumberOfFaces()
                   << " this->getName() " << this->getName(); );
   if( faceIndex < this->numberOfNxFaces )
   {
      nx = 1;
      ny = 0;
      const IndexType aux = this->getDimensions().x() + 1;
      return CoordinatesType( faceIndex % aux, faceIndex / aux );
   }
   else
   {
      nx = 0;
      ny = 1;
      const IndexType i = faceIndex - this->numberOfNxFaces;
      const IndexType& aux = this->getDimensions().x();
      return CoordinatesType( i % aux, i / aux );
   }
}







template< typename Real,
          typename Device,
          typename Index >
Index tnlGrid< 2, Real, Device, Index > :: getEdgeIndex( const Index i,
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
   if( dy == 0 )
      return j * ( this -> dimensions. x() + 1 ) + i + ( 1 + dx ) / 2;
   return this -> dimensions. y() * ( this -> dimensions. x() + 1 ) +
            ( j + ( 1 + dy ) / 2 ) * this -> dimensions. x() + i;
}

template< typename Real,
          typename Device,
          typename Index >
   template< int dx, int dy >
Index tnlGrid< 2, Real, Device, Index > :: getVertexIndex( const Index i,
                                                                     const Index j ) const
{
   tnlAssert( i < dimensions. x(),
              cerr << "Index i ( " << i
                   << " ) is out of range ( " << dimensions. x()
                   << " ) in tnlGrid " << this -> getName(); );
   tnlAssert( j < dimensions. y(),
              cerr << "Index j ( " << j
                   << " ) is out of range ( " << dimensions. y()
                   << " ) in tnlGrid " << this -> getName(); );
   tnlAssert( ( dy == 1 || dy == -1 ) ||
              ( dx == 1 || dx == -1 ),
              cerr << "dx = " << dx << ", dy = " << dy << endl;);
   return ( j + ( 1 + dy ) / 2 ) * ( this -> dimensions. x() + 1 ) + i + ( 1 + dx ) / 2;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 2, Real, Device, Index > :: refresh()
{
   /*if( GeometryType :: ElementsMeasureStorage :: enabled )
      for( IndexType j = 0; j < dimensions. y(); j ++ )
         for( IndexType i = 0; i < dimensions. x(); i ++ )
            elementsMeasure[ getElementIndex( i, j ) ] = geometry. getElementMeasure( CoordinatesType( i, j ) );

   if( GeometryType :: DualElementsMeasureStorage :: enabled )
      for( IndexType j = 1; j < dimensions. y() - 1; j ++ )
         for( IndexType i = 1; i < dimensions. x() - 1; i ++ )
         {
            dualElementsMeasure[ getEdgeIndex( i, j,  1,  0 ) ] = geometry. getDualElementMeasure<  1,  0 >( CoordinatesType( i, j ) );
            dualElementsMeasure[ getEdgeIndex( i, j, -1,  0 ) ] = geometry. getDualElementMeasure< -1,  0 >( CoordinatesType( i, j ) );
            dualElementsMeasure[ getEdgeIndex( i, j,  0,  1 ) ] = geometry. getDualElementMeasure<  0,  1 >( CoordinatesType( i, j ) );
            dualElementsMeasure[ getEdgeIndex( i, j,  0, -1 ) ] = geometry. getDualElementMeasure<  0, -1 >( CoordinatesType( i, j ) );
         }

   if( GeometryType :: EdgeNormalsStorage :: enabled )
      for( IndexType j = 0; j < dimensions. y(); j ++ )
         for( IndexType i = 0; i < dimensions. x(); i ++ )
         {
            geometry. getEdgeNormal<  1,  0 >( CoordinatesType( i, j ), edgeNormals[ getEdgeIndex( i, j,  1,  0 ) ] );
            geometry. getEdgeNormal< -1,  0 >( CoordinatesType( i, j ), edgeNormals[ getEdgeIndex( i, j, -1,  0 ) ] );
            geometry. getEdgeNormal<  0,  1 >( CoordinatesType( i, j ), edgeNormals[ getEdgeIndex( i, j,  0,  1 ) ] );
            geometry. getEdgeNormal<  0, -1 >( CoordinatesType( i, j ), edgeNormals[ getEdgeIndex( i, j,  0, -1 ) ] );
            edgeNormals[ getEdgeIndex( i, j, -1, 0 ) ] *= ( RealType ) -1.0;
            edgeNormals[ getEdgeIndex( i, j, 0, -1 ) ] *= ( RealType ) -1.0;
         }

   if( GeometryType :: VerticesStorage :: enabled )
      for( IndexType j = 0; j < dimensions. y(); j ++ )
      {
         for( IndexType i = 0; i < dimensions. x(); i ++ )
         {
            const CoordinatesType c( i, j );
            geometry. template getVertex<  1,  1 >( c, origin, vertices[ getVertexIndex<  1,  1 >( i, j ) ] );
            geometry. template getVertex< -1,  1 >( c, origin, vertices[ getVertexIndex< -1,  1 >( i, j ) ] );
            geometry. template getVertex< -1, -1 >( c, origin, vertices[ getVertexIndex< -1, -1 >( i, j ) ] );
            geometry. template getVertex<  1, -1 >( c, origin, vertices[ getVertexIndex<  1, -1 >( i, j ) ] );
            geometry. template getVertex<  0,  0 >( c, origin, elementCenters[ getElementIndex( i, j ) ] );
            //cout << "( " << elementCenters[ getElementIndex( i, j ) ] << " ), ";
         }
         //cout << endl;
      }*/
}

template< typename Real,
          typename Device,
          typename Index >
void tnlGrid< 2, Real, Device, Index > :: getElementCoordinates( const Index element,
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
          typename Index >
Index tnlGrid< 2, Real, Device, Index > :: getElementNeighbour( const Index Element,
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
          typename Index >
Index tnlGrid< 2, Real, Device, Index > :: getDofs() const
{
   return this -> dofs;
};

template< typename Real,
          typename Device,
          typename Index >
Index tnlGrid< 2, Real, Device, Index > :: getNumberOfEdges() const
{
   return 2 * this -> dimensions. x() * this -> dimensions. y() +
          this -> dimensions. x() + this -> dimensions. y();
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlGrid< 2, Real, Device, Index > :: getNumberOfVertices() const
{
   return ( this -> dimensions. x() + 1 ) * ( this -> dimensions. y() + 1 );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vertex >
void tnlGrid< 2, Real, Device, Index > :: getElementCenter( const CoordinatesType& coordinates,
                                                                      Vertex& center ) const
{
      //geometry. getElementCenter( origin, coordinates, center );
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlGrid< 2, Real, Device, Index > :: getElementMeasure( const CoordinatesType& coordinates ) const
{
   /*if( GeometryType :: ElementsMeasureStorage :: enabled )
      return elementsMeasure[ getElementIndex( coordinates. x(), coordinates. y() ) ];
   return geometry. getElementMeasure( coordinates );*/
}

template< typename Real,
          typename Device,
          typename Index >
   template< int dx, int dy >
Real tnlGrid< 2, Real, Device, Index > :: getDualElementMeasure( const CoordinatesType& coordinates ) const
{
   /*if( GeometryType :: DualElementsMeasureStorage :: enabled )
      return dualElementsMeasure[ getEdgeIndex( coordinates. x(), coordinates. y(), dx, dy ) ];
   return geometry. getDualElementMeasure< dx, dy >( coordinates );*/
}

template< typename Real,
          typename Device,
          typename Index >
template< int dx, int dy >
void tnlGrid< 2, Real, Device, Index > :: getEdgeNormal( const CoordinatesType& coordinates,
                                                                   VertexType& normal ) const
{
   /*if( GeometryType :: EdgeNormalsStorage :: enabled )
   {
      normal = edgeNormals[ getEdgeIndex( coordinates. x(), coordinates. y(), dx, dy ) ];
      if( dx == -1 || dy == -1 )
         normal *= ( RealType ) -1.0;
   }
   else
      return geometry. getEdgeNormal< dx, dy >( coordinates, normal );*/
}

template< typename Real,
          typename Device,
          typename Index >
   template< int dx, int dy, typename Vertex >
void tnlGrid< 2, Real, Device, Index > :: getVertex( const CoordinatesType& elementCoordinates,
                                                               Vertex& vertex ) const
{
   /*tnlAssert( elementCoordinates. x() >= 0 &&
              elementCoordinates. x() < this -> dimensions. x() &&
              elementCoordinates. y() >= 0 &&
              elementCoordinates. y() < this -> dimensions. y(),
              cerr << "elementCoordinates = " << elementCoordinates << endl; );
   if( GeometryType :: VerticesStorage :: enabled )
      if( dx == 0 && dy == 0 )
         vertex = this -> elementCenters[ this -> getElementIndex( elementCoordinates. x(),
                                                                   elementCoordinates. y() ) ];
      else
         vertex = this -> vertices[ this -> getVertexIndex< dx, dy >( elementCoordinates. x(),
                                                                      elementCoordinates. y() )];
   else
      return geometry. getVertex< dx, dy >( elementCoordinates,
                                            this -> origin,
                                            vertex );*/
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
      typename GridFunction::RealType
         tnlGrid< 2, Real, Device, Index >::getAbsMax( const GridFunction& f ) const
{
   return f.absMax();
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
      typename GridFunction::RealType
         tnlGrid< 2, Real, Device, Index >::getLpNorm( const GridFunction& f1,
                                                                 const typename GridFunction::RealType& p ) const
{
   typename GridFunction::RealType lpNorm( 0.0 );
   for( IndexType j = 0; j < getDimensions(). y(); j++ )
      for( IndexType i = 0; i < getDimensions(). x(); i++ )
      {
         IndexType c = this->getElementIndex( i, j );
         lpNorm += pow( tnlAbs( f1[ c ] ), p ) *
            this->getElementMeasure( CoordinatesType( i, j ) );
      }
   return pow( lpNorm, 1.0/p );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
      typename GridFunction::RealType
         tnlGrid< 2, Real, Device, Index >::getDifferenceAbsMax( const GridFunction& f1,
                                                                           const GridFunction& f2 ) const
{
   return f1.differenceAbsMax( f2 );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename GridFunction >
      typename GridFunction::RealType
         tnlGrid< 2, Real, Device, Index >::getDifferenceLpNorm( const GridFunction& f1,
                                                                           const GridFunction& f2,
                                                                           const typename GridFunction::RealType& p ) const
{
   typename GridFunction::RealType lpNorm( 0.0 );
   for( IndexType j = 0; j < getDimensions(). y(); j++ )
      for( IndexType i = 0; i < getDimensions(). x(); i++ )
      {
         IndexType c = this->getElementIndex( i, j );
         lpNorm += pow( tnlAbs( f1[ c ] - f2[ c ] ), p ) *
            this->getElementMeasure( CoordinatesType( i, j ) );
      }
   return pow( lpNorm, 1.0/p );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlGrid< 2, Real, Device, Index > :: save( tnlFile& file ) const
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
   return true;
};

template< typename Real,
          typename Device,
          typename Index >
bool tnlGrid< 2, Real, Device, Index > :: load( tnlFile& file )
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
   if( ! this -> setDimensions( dim ) )
   {
      cerr << "I am not able to allocate the loaded grid." << endl;
      return false;
   }
   this -> refresh();
   return true;
};

template< typename Real,
          typename Device,
          typename Index >
bool tnlGrid< 2, Real, Device, Index > :: save( const tnlString& fileName ) const
{
   return tnlObject :: save( fileName );
};

template< typename Real,
           typename Device,
           typename Index >
bool tnlGrid< 2, Real, Device, Index > :: load( const tnlString& fileName )
{
   return tnlObject :: load( fileName );
};

template< typename Real,
           typename Device,
           typename Index >
bool tnlGrid< 2, Real, Device, Index > :: writeMesh( const tnlString& fileName,
                                                               const tnlString& format ) const
{
   fstream file;
   file. open( fileName. getString(), ios :: out );
   if( ! file )
   {
      cerr << "I am not able to open the file " << fileName << "." << endl;
      return false;
   }
   if( format == "asymptote" )
   {
      file << "size( "
           << this -> getProportions(). x() << "cm , "
           << this -> getProportions(). y() << "cm );"
           << endl << endl;
      VertexType v;
      for( Index j = 0; j < this -> dimensions. y(); j ++ )
      {
         file << "draw( ";
         this -> getVertex< -1, -1 >( CoordinatesType( 0, j ), v );
         file << "( " << v. x() << ", " << v. y() << " )";
         for( Index i = 0; i < this -> dimensions. x(); i ++ )
         {
            this -> getVertex< 1, -1 >( CoordinatesType( i, j ), v );
            file << "--( " << v. x() << ", " << v. y() << " )";
         }
         file << " );" << endl;
      }
      file << endl;
      for( Index i = 0; i < this -> dimensions. x(); i ++ )
      {
         file << "draw( ";
         this -> getVertex< -1, -1 >( CoordinatesType( i, 0 ), v );
         file << "( " << v. x() << ", " << v. y() << " )";
         for( Index j = 0; j < this -> dimensions. y(); j ++ )
         {
            this -> getVertex< -1, 1 >( CoordinatesType( i, j ), v );
            file << "--( " << v. x() << ", " << v. y() << " )";
         }
         file << " );" << endl;
      }
      file << endl;
      for( Index i = 0; i < this -> dimensions. x(); i ++ )
         for( Index j = 0; j < this -> dimensions. y(); j ++ )
         {
            this -> getVertex< 0, 0 >( CoordinatesType( i, j ), v );
            file << "label( scale(0.33) * Label( \"$" << setprecision( 3 ) << this -> getElementMeasure( CoordinatesType( i, j ) ) << setprecision( 8 )
                 << "$\" ), ( " << v. x() << ", " << v. y() << " ), S );" << endl;
         }

      for( Index i = 0; i < this -> dimensions. x(); i ++ )
         for( Index j = 0; j < this -> dimensions. y(); j ++ )
         {
            VertexType v1, v2, c;

            /****
             * East edge normal
             */
            this -> getVertex< 1, -1 >( CoordinatesType( i, j ), v1 );
            this -> getVertex< 1, 1 >( CoordinatesType( i, j ), v2 );
            c = ( ( Real ) 0.5 ) * ( v1 + v2 );
            this -> getEdgeNormal< 1, 0 >( CoordinatesType( i, j ), v );
            v *= 0.5;
            file << "draw( ( " << c. x() << ", " << c. y() << " )--( "
                 << c. x() + v. x() << ", " << c.y() + v. y() << " ), Arrow(size=1mm),p=green);" << endl;

            /****
             * West edge normal
             */
            this -> getVertex< -1, -1 >( CoordinatesType( i, j ), v1 );
            this -> getVertex< -1, 1 >( CoordinatesType( i, j ), v2 );
            c = ( ( Real ) 0.5 ) * ( v1 + v2 );
            this -> getEdgeNormal< -1, 0 >( CoordinatesType( i, j ), v );
            v *= 0.5;
            file << "draw( ( " << c. x() << ", " << c. y() << " )--( "
                 << c. x() + v. x() << ", " << c.y() + v. y() << " ), Arrow(size=1mm),p=blue);" << endl;

            /****
             * North edge normal
             */
            this -> getVertex< 1, 1 >( CoordinatesType( i, j ), v1 );
            this -> getVertex< -1, 1 >( CoordinatesType( i, j ), v2 );
            c = ( ( Real ) 0.5 ) * ( v1 + v2 );
            this -> getEdgeNormal< 0, 1 >( CoordinatesType( i, j ), v );
            v *= 0.5;
            file << "draw( ( " << c. x() << ", " << c. y() << " )--( "
                 << c. x() + v. x() << ", " << c.y() + v. y() << " ), Arrow(size=1mm),p=green);" << endl;

            /****
             * South edge normal
             */
            this -> getVertex< 1, -1 >( CoordinatesType( i, j ), v1 );
            this -> getVertex< -1, -1 >( CoordinatesType( i, j ), v2 );
            c = ( ( Real ) 0.5 ) * ( v1 + v2 );
            this -> getEdgeNormal< 0, -1 >( CoordinatesType( i, j ), v );
            v *= 0.5;
            file << "draw( ( " << c. x() << ", " << c. y() << " )--( "
                 << c. x() + v. x() << ", " << c.y() + v. y() << " ), Arrow(size=1mm),p=blue);" << endl;
         }

   }
}

template< typename Real,
           typename Device,
           typename Index >
   template< typename MeshFunction >
bool tnlGrid< 2, Real, Device, Index > :: write( const MeshFunction& function,
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
      for( IndexType j = 0; j < getDimensions(). y(); j++ )
      {
         for( IndexType i = 0; i < getDimensions(). x(); i++ )
         {
            VertexType v;
            this -> getVertex< 0, 0 >( CoordinatesType( i, j ), v );
            tnlGnuplotWriter::write( file,  v );
            tnlGnuplotWriter::write( file,  function[ this -> getElementIndex( i, j ) ] );
            file << endl;
         }
         file << endl;
      }

   file. close();
   return true;
}


#endif /* TNLGRID2D_IMPL_H_ */
