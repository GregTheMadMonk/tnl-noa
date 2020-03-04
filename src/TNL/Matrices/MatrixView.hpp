/***************************************************************************
                          MatrixView.hpp  -  description
                             -------------------
    begin                : Dec 28, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/Matrix.h>
#include <TNL/Assert.h>
#include <TNL/Cuda/LaunchHelpers.h>
#include <TNL/Cuda/MemoryHelpers.h>
#include <TNL/Cuda/SharedMemory.h>

namespace TNL {
namespace Matrices {

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
MatrixView< Real, Device, Index >::
MatrixView()
: rows( 0 ),
  columns( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
MatrixView< Real, Device, Index >::
MatrixView( const IndexType rows_,
            const IndexType columns_,
            const ValuesView& values_ )
 : rows( rows_ ), columns( columns_ ), values( values_ )
{
}

template< typename Real,
          typename Device,
          typename Index >
Index
MatrixView< Real, Device, Index >::
getAllocatedElementsCount() const
{
   return this->values.getSize();
}

template< typename Real,
          typename Device,
          typename Index >
Index
MatrixView< Real, Device, Index >::
getNumberOfNonzeroMatrixElements() const
{
   const auto values_view = this->values.getConstView();
   auto fetch = [=] __cuda_callable__ ( const IndexType i ) -> IndexType {
      return ( values_view[ i ] != 0.0 );
   };
   return Algorithms::Reduction< DeviceType >::reduce( this->values.getSize(), std::plus<>{}, fetch, 0 );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index MatrixView< Real, Device, Index >::getRows() const
{
   return this->rows;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index MatrixView< Real, Device, Index >::getColumns() const
{
   return this->columns;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const typename MatrixView< Real, Device, Index >::ValuesView&
MatrixView< Real, Device, Index >::
getValues() const
{
   return this->values;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
typename MatrixView< Real, Device, Index >::ValuesView&
MatrixView< Real, Device, Index >::
getValues()
{
   return this->values;
}
template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
MatrixView< Real, Device, Index >&
MatrixView< Real, Device, Index >::
operator=( const MatrixView& view )
{
   rows = view.rows;
   columns = view.columns;
   values.bind( view.values );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename MatrixT >
bool MatrixView< Real, Device, Index >::operator == ( const MatrixT& matrix ) const
{
   if( this->getRows() != matrix.getRows() ||
       this->getColumns() != matrix.getColumns() )
      return false;
   for( IndexType row = 0; row < this->getRows(); row++ )
      for( IndexType column = 0; column < this->getColumns(); column++ )
         if( this->getElement( row, column ) != matrix.getElement( row, column ) )
            return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename MatrixT >
bool MatrixView< Real, Device, Index >::operator != ( const MatrixT& matrix ) const
{
   return ! operator == ( matrix );
}

template< typename Real,
          typename Device,
          typename Index >
void MatrixView< Real, Device, Index >::save( File& file ) const
{
   Object::save( file );
   file.save( &this->rows );
   file.save( &this->columns );
   file << this->values;
}

template< typename Real,
          typename Device,
          typename Index >
void MatrixView< Real, Device, Index >::load( File& file )
{
   Object::load( file );
   file.load( &this->rows );
   file.load( &this->columns );
   file >> this->values;
}

template< typename Real,
          typename Device,
          typename Index >
void MatrixView< Real, Device, Index >::print( std::ostream& str ) const
{
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Index&
MatrixView< Real, Device, Index >::
getNumberOfColors() const
{
   return this->numberOfColors;
}

template< typename Real,
          typename Device,
          typename Index >
void
MatrixView< Real, Device, Index >::
computeColorsVector(Containers::Vector<Index, Device, Index> &colorsVector)
{
    for( IndexType i = this->getRows() - 1; i >= 0; i-- )
    {
        // init color array
        Containers::Vector< Index, Device, Index > usedColors;
        usedColors.setSize( this->numberOfColors );
        for( IndexType j = 0; j < this->numberOfColors; j++ )
            usedColors.setElement( j, 0 );

        // find all colors used in given row
        for( IndexType j = i + 1; j < this->getColumns(); j++ )
             if( this->getElement( i, j ) != 0.0 )
                 usedColors.setElement( colorsVector.getElement( j ), 1 );

        // find unused color
        bool found = false;
        for( IndexType j = 0; j < this->numberOfColors; j++ )
            if( usedColors.getElement( j ) == 0 )
            {
                colorsVector.setElement( i, j );
                found = true;
                break;
            }
        if( !found )
        {
            colorsVector.setElement( i, this->numberOfColors );
            this->numberOfColors++;
        }
    }
}

} // namespace Matrices
} // namespace TNL