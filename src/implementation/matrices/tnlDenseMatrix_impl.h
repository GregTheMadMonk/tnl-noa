/***************************************************************************
                          tnlDenseMatrix_impl.h  -  description
                             -------------------
    begin                : Nov 29, 2013
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

#ifndef TNLDENSEMATRIX_IMPL_H_
#define TNLDENSEMATRIX_IMPL_H_

#include <core/tnlAssert.h>
#include <matrices/tnlDenseMatrix.h>

template< typename Real,
          typename Device,
          typename Index >
tnlDenseMatrix< Real, Device, Index >::tnlDenseMatrix()
{
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlDenseMatrix< Real, Device, Index >::getType()
{
   return tnlString( "tnlDenseMatrix< " ) +
          tnlString( getParameterType< RealType >() ) + ", " +
          tnlString( Device :: getDeviceType() ) + ", " +
          tnlString( getParameterType< IndexType >() ) + " >";
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlDenseMatrix< Real, Device, Index >::getTypeVirtual() const
{
   return this -> getType();
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlDenseMatrix< Real, Device, Index >::setDimensions( const IndexType rows,
                                                           const IndexType columns )
{
   if( ! tnlMatrix< Real, Device, Index >::setDimensions( rows, columns ) ||
       ! this->values.setSize( rows * columns ) )
     return false;
   this->values.setValue( 0.0 );
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlDenseMatrix< Real, Device, Index >::setLike( const tnlDenseMatrix< Real2, Device2, Index2 >& matrix )
{
   return this->setDimensions( matrix.getRows(), matrix.getColumns() );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlDenseMatrix< Real, Device, Index >::setRowLengths( const RowLengthsVector& rowLengths )
{
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlDenseMatrix< Real, Device, Index >::getRowLength( const IndexType row ) const
{
   return this->getColumns();
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlDenseMatrix< Real, Device, Index >::getNumberOfMatrixElements() const
{
   return this->getRows() * this->getColumns();
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlDenseMatrix< Real, Device, Index >::getNumberOfNonzeroMatrixElements() const
{
   IndexType nonzeroElements( 0 );
   for( IndexType row = 0; row < this->getRows(); row++ )
      for( IndexType column = 0; column < this->getColumns(); column++ )
         if( this->getElement( row, column ) != 0 )
            nonzeroElements++;
   return nonzeroElements;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlDenseMatrix< Real, Device, Index >::reset()
{
   tnlMatrix< Real, Device, Index >::reset();
   this->values.reset();
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlDenseMatrix< Real, Device, Index >::setElementFast( const IndexType row,
                                                            const IndexType column,
                                                            const RealType& value )
{
   this->values.operator[]( this->getElementIndex( row, column ) ) = value;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlDenseMatrix< Real, Device, Index >::setElement( const IndexType row,
                                                        const IndexType column,
                                                        const RealType& value )
{
   this->values.setElement( this->getElementIndex( row, column ), value );
   return true;
}


template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlDenseMatrix< Real, Device, Index >::addElementFast( const IndexType row,
                                                            const IndexType column,
                                                            const RealType& value,
                                                            const RealType& thisElementMultiplicator )
{
   const IndexType elementIndex = this->getElementIndex( row, column );
   if( thisElementMultiplicator == 1.0 )
      this->values.operator[]( elementIndex ) += value;
   else
      this->values.operator[]( elementIndex ) =
         thisElementMultiplicator * this->values.operator[]( elementIndex ) + value;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlDenseMatrix< Real, Device, Index >::addElement( const IndexType row,
                                                        const IndexType column,
                                                        const RealType& value,
                                                        const RealType& thisElementMultiplicator )
{
   const IndexType elementIndex = this->getElementIndex( row, column );
   if( thisElementMultiplicator == 1.0 )
      this->values.setElement( elementIndex,
                               this->values.getElement( elementIndex ) + value );
   else
      this->values.setElement( elementIndex,
                               thisElementMultiplicator * this->values.getElement( elementIndex ) + value );
}


template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlDenseMatrix< Real, Device, Index >::setRowFast( const IndexType row,
                                                        const IndexType* columns,
                                                        const RealType* values,
                                                        const IndexType elements )
{
   tnlAssert( elements <= this->getColumns(),
            cerr << " elements = " << elements
                 << " this->columns = " << this->getColumns()
                 << " this->getName() = " << this->getName() );
   for( IndexType i = 0; i < elements; i++ )
      this->setElementFast( row, columns[ i ], values[ i ] );
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlDenseMatrix< Real, Device, Index >::setRow( const IndexType row,
                                                    const IndexType* columns,
                                                    const RealType* values,
                                                    const IndexType elements )
{
   tnlAssert( elements <= this->getColumns(),
            cerr << " elements = " << elements
                 << " this->columns = " << this->getColumns()
                 << " this->getName() = " << this->getName() );
   for( IndexType i = 0; i < elements; i++ )
      this->setElement( row, columns[ i ], values[ i ] );
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlDenseMatrix< Real, Device, Index >::addRowFast( const IndexType row,
                                                        const IndexType* columns,
                                                        const RealType* values,
                                                        const IndexType elements,
                                                        const RealType& thisRowMultiplicator )
{
   tnlAssert( elements <= this->columns,
            cerr << " elements = " << elements
                 << " this->columns = " << this->columns
                 << " this->getName() = " << this->getName() );
   for( IndexType i = 0; i < elements; i++ )
      this->setElementFast( row, columns[ i ],
                            thisRowMultiplicator * this->getElementFast( row, columns[ i ] ) + values[ i ] );
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlDenseMatrix< Real, Device, Index >::addRow( const IndexType row,
                                                    const IndexType* columns,
                                                    const RealType* values,
                                                    const IndexType elements,
                                                    const RealType& thisRowMultiplicator )
{
   tnlAssert( elements <= this->columns,
            cerr << " elements = " << elements
                 << " this->columns = " << this->columns
                 << " this->getName() = " << this->getName() );
   for( IndexType i = 0; i < elements; i++ )
      this->setElement( row, columns[ i ],
                        thisRowMultiplicator * this->getElement( row, columns[ i ] ) + values[ i ] );
   return true;
}


template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Real tnlDenseMatrix< Real, Device, Index >::getElementFast( const IndexType row,
                                                            const IndexType column ) const
{
   return this->values.operator[]( this->getElementIndex( row, column ) );
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlDenseMatrix< Real, Device, Index >::getElement( const IndexType row,
                                                        const IndexType column ) const
{
   return this->values.getElement( this->getElementIndex( row, column ) );
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
void tnlDenseMatrix< Real, Device, Index >::getRowFast( const IndexType row,
                                                        IndexType* columns,
                                                        RealType* values ) const
{
   for( IndexType i = 0; i < this->getColumns(); i++ )
   {
      columns[ i ] = i;
      values[ i ] = this->getElementFast( row, i );
   }
}

template< typename Real,
          typename Device,
          typename Index >
void tnlDenseMatrix< Real, Device, Index >::getRow( const IndexType row,
                                                    IndexType* columns,
                                                    RealType* values ) const
{
   for( IndexType i = 0; i < this->getColumns(); i++ )
   {
      columns[ i ] = i;
      values[ i ] = this->getElement( row, i );
   }
}


template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
typename Vector::RealType tnlDenseMatrix< Real, Device, Index >::rowVectorProduct( const IndexType row,
                                                                                   const Vector& vector ) const
{
   RealType sum( 0.0 );
   for( IndexType column = 0; column < this->getColumns(); column++ )
      sum += this->getElement( row, column ) * vector.getElement( column );
   return sum;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
void tnlDenseMatrix< Real, Device, Index >::vectorProduct( const Vector& inVector,
                                                           Vector& outVector ) const
{
   tnlAssert( this->getColumns() == inVector.getSize(),
            cerr << "Matrix columns: " << this->getColumns() << endl
                 << "Matrix name: " << this->getName() << endl
                 << "Vector size: " << inVector.getSize() << endl
                 << "Vector name: " << inVector.getName() << endl );
   tnlAssert( this->getRows() == outVector.getSize(),
               cerr << "Matrix rows: " << this->getRows() << endl
                    << "Matrix name: " << this->getName() << endl
                    << "Vector size: " << outVector.getSize() << endl
                    << "Vector name: " << outVector.getName() << endl );

   for( IndexType row = 0; row < this->getRows(); row++ )
      outVector[ row ] = rowVectorProduct( row, inVector );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Matrix >
void tnlDenseMatrix< Real, Device, Index >::addMatrix( const Matrix& matrix,
                                                       const RealType& matrixMultiplicator,
                                                       const RealType& thisMatrixMultiplicator )
{
   tnlAssert( this->getColumns() == matrix.getColumns() &&
              this->getRows() == matrix.getRows(),
            cerr << "This matrix columns: " << this->getColumns() << endl
                 << "This matrix rows: " << this->getRows() << endl
                 << "This matrix name: " << this->getName() << endl
                 << "That matrix columns: " << matrix.getColumns() << endl
                 << "That matrix rows: " << matrix.getRows() << endl
                 << "That matrix name: " << matrix.getName() << endl );

   if( thisMatrixMultiplicator == 1.0 )
   {
      for( IndexType row = 0; row < this->getRows(); row++ )
         for( IndexType column = 0; column < this->getColumns(); column++ )
            this->operator()( row, column ) += matrixMultiplicator*matrix( row, column );
   }
   else
   {
      for( IndexType row = 0; row < this->getRows(); row++ )
         for( IndexType column = 0; column < this->getColumns(); column++ )
            this->operator()( row, column ) =
                thisMatrixMultiplicator * this->operator()( row, column) +
                   matrixMultiplicator * matrix( row, column );
   }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Matrix1, typename Matrix2, int tileDim >
void tnlDenseMatrix< Real, Device, Index >::getMatrixProduct( const Matrix1& matrix1,
                                                              const Matrix2& matrix2,
                                                              const RealType& matrix1Multiplicator,
                                                              const RealType& matrix2Multiplicator )
{
   tnlAssert( matrix1.getColumns() == matrix2.getRows() &&
              this->getRows() == matrix1.getRows() &&
              this->getColumns() == matrix2.getColumns(),
            cerr << "This matrix columns: " << this->getColumns() << endl
                 << "This matrix rows: " << this->getRows() << endl
                 << "This matrix name: " << this->getName() << endl
                 << "Matrix1 columns: " << matrix1.getColumns() << endl
                 << "Matrix1 rows: " << matrix1.getRows() << endl
                 << "Matrix1 name: " << matrix1.getName() << endl
                 << "Matrix2 columns: " << matrix2.getColumns() << endl
                 << "Matrix2 rows: " << matrix2.getRows() << endl
                 << "Matrix2 name: " << matrix2.getName() << endl );

   for( IndexType i = 0; i < this->getRows(); i += tileDim )
      for( IndexType j = 0; j < this->getColumns(); j += tileDim )
      {
         const IndexType tileRows = Min( tileDim, this->getRows() - i );
         const IndexType tileColumns = Min( tileDim, this->getColumns() - j );
         for( IndexType i1 = i; i1 < i + tileRows; i1++ )
            for( IndexType j1 = j; j1 < j + tileColumns; j1++ )
               this->operator()( i1, j1 ) = 0.0;

         for( IndexType k = 0; k < matrix1.getColumns(); k += tileDim )
         {
            const IndexType lastK = Min( k + tileDim, matrix1.getColumns() );
            for( IndexType i1 = 0; i1 < tileRows; i1++ )
               for( IndexType j1 = 0; j1 < tileColumns; j1++ )
                  for( IndexType k1 = k; k1 < lastK; k1++ )
                     this->operator()( i + i1, j + j1 ) +=
                        matrix1( i + i1, k1 ) * matrix2( k1, j + j1 );
         }
      }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Matrix, int tileDim >
void tnlDenseMatrix< Real, Device, Index >::getTransposition( const Matrix& matrix,
                                                              const RealType& matrixMultiplicator )
{
   tnlAssert( this->getColumns() == matrix.getRows() &&
              this->getRows() == matrix.getColumns(),
               cerr << "This matrix columns: " << this->getColumns() << endl
                    << "This matrix rows: " << this->getRows() << endl
                    << "This matrix name: " << this->getName() << endl
                    << "That matrix columns: " << matrix.getColumns() << endl
                    << "That matrix rows: " << matrix.getRows() << endl
                    << "That matrix name: " << matrix.getName() << endl );

   const IndexType& rows = matrix.getRows();
   const IndexType& columns = matrix.getColumns();
   for( IndexType i = 0; i < rows; i += tileDim )
      for( IndexType j = 0; j < columns; j += tileDim )
         for( IndexType k = i; k < i + tileDim && k < rows; k++ )
            for( IndexType l = j; l < j + tileDim && l < columns; l++ )
               this->operator()( l, k ) = matrix( k, l );

}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
void tnlDenseMatrix< Real, Device, Index >::performSORIteration( const Vector& b,
                                                                 const IndexType row,
                                                                 Vector& x,
                                                                 const RealType& omega ) const
{
   RealType sum( 0.0 );
   for( IndexType i = 0; i < this->getColumns(); i++ )
      sum += this->operator()( row, i ) * x[ i ];
   x[ row ] += omega / this->operator()( row, row )( b[ row ] - sum );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlDenseMatrix< Real, Device, Index >::save( const tnlString& fileName ) const
{
   return tnlObject::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlDenseMatrix< Real, Device, Index >::load( const tnlString& fileName )
{
   return tnlObject::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlDenseMatrix< Real, Device, Index >::save( tnlFile& file ) const
{
   if( ! tnlMatrix< Real, Device, Index >::save( file )  )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlDenseMatrix< Real, Device, Index >::load( tnlFile& file )
{
   if( ! tnlMatrix< Real, Device, Index >::load( file ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlDenseMatrix< Real, Device, Index >::print( ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";
      for( IndexType column = 0; column < this->getColumns(); column++ )
         str << " Col:" << column << "->" << this->getElement( row, column ) << "\t";
      str << endl;
   }
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Index tnlDenseMatrix< Real, Device, Index >::getElementIndex( const IndexType row,
                                                              const IndexType column ) const
{
   if( Device::getDevice() == tnlHostDevice )
      return row * this->columns + column;
   if( Device::getDevice() == tnlCudaDevice)
      return column * this->rows + row;
}


#endif /* TNLDENSEMATRIX_IMPL_H_ */
