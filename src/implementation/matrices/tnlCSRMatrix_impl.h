/***************************************************************************
                          tnlCSRMatrix_impl.h  -  description
                             -------------------
    begin                : Dec 10, 2013
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

#ifndef TNLCSRMATRIX_IMPL_H_
#define TNLCSRMATRIX_IMPL_H_

#include <matrices/tnlCSRMatrix.h>
#include <core/vectors/tnlVector.h>
#include <core/mfuncs.h>

template< typename Real,
          typename Device,
          typename Index >
tnlCSRMatrix< Real, Device, Index >::tnlCSRMatrix()
{
};

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlCSRMatrix< Real, Device, Index >::getType()
{
   return tnlString( "tnlCSRMatrix< ") +
          tnlString( GetParameterType( Real( 0.0 ) ) ) +
          tnlString( ", " ) +
          Device :: getDeviceType() +
          tnlString( " >" );
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlCSRMatrix< Real, Device, Index >::getTypeVirtual() const
{
   return this->getType();
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlCSRMatrix< Real, Device, Index >::setDimensions( const IndexType rows,
                                                         const IndexType columns )
{
   if( ! tnlSparseMatrix< Real, Device, Index >::setDimensions( rows, columns ) ||
       ! this->rowPointers.setSize( this->rows + 1 ) )
      return false;
   this->rowPointers.setValue( 0 );
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlCSRMatrix< Real, Device, Index >::setRowLengths( const RowLengthsVector& rowLengths )
{
   /****
    * Compute the rows pointers. The last one is
    * the end of the last row and so it says the
    * necessary length of the vectors this->values
    * and this->columnIndexes.
    */
   for( IndexType i = 0; i < this->rows; i++ )
      this->rowPointers[ i ] = rowLengths[ i ];
   this->rowPointers[ this->rows ] = 0;
   this->rowPointers.computeExclusivePrefixSum();

   /****
    * Allocate values and column indexes
    */
   if( ! this->values.setSize( this->rowPointers[ this->rows ] ) ||
       ! this->columnIndexes.setSize( this->rowPointers[ this->rows ] ) )
      return false;
   this->columnIndexes.setValue( this->columns );
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlCSRMatrix< Real, Device, Index >::getRowLength( const IndexType row ) const
{
   return this->rowPointers[ row + 1 ] - this->rowPointers[ row ];
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlCSRMatrix< Real, Device, Index >::setLike( const tnlCSRMatrix< Real2, Device2, Index2 >& matrix )
{
   if( ! tnlSparseMatrix< Real, Device, Index >::setLike( matrix ) ||
       ! this->rowPointers.setLike( matrix.rowPointers ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlCSRMatrix< Real, Device, Index >::reset()
{
   tnlSparseMatrix< Real, Device, Index >::reset();
   this->rowPointers.reset();
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlCSRMatrix< Real, Device, Index >::operator == ( const tnlCSRMatrix< Real2, Device2, Index2 >& matrix ) const
{
   tnlAssert( this->getRows() == matrix.getRows() &&
              this->getColumns() == matrix.getColumns(),
              cerr << "this->getRows() = " << this->getRows()
                   << " matrix.getRows() = " << matrix.getRows()
                   << " this->getColumns() = " << this->getColumns()
                   << " matrix.getColumns() = " << matrix.getColumns()
                   << " this->getName() = " << this->getName()
                   << " matrix.getName() = " << matrix.getName() );
   // TODO: implement this
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool tnlCSRMatrix< Real, Device, Index >::operator != ( const tnlCSRMatrix< Real2, Device2, Index2 >& matrix ) const
{
   return ! ( ( *this ) == matrix );
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlCSRMatrix< Real, Device, Index >::setElementFast( const IndexType row,
                                                          const IndexType column,
                                                          const Real& value )
{
   return this->addElementFast( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlCSRMatrix< Real, Device, Index >::setElement( const IndexType row,
                                                      const IndexType column,
                                                      const Real& value )
{
   return this->addElementFast( row, column, value, 0.0 );
}


template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlCSRMatrix< Real, Device, Index >::addElementFast( const IndexType row,
                                                          const IndexType column,
                                                          const RealType& value,
                                                          const RealType& thisElementMultiplicator )
{
   tnlAssert( row >= 0 && row < this->rows &&
              column >= 0 && column <= this->rows,
              cerr << " row = " << row
                   << " column = " << column
                   << " this->rows = " << this->rows
                   << " this->columns = " << this-> columns );

   IndexType elementPtr = this->rowPointers[ row ];
   const IndexType rowEnd = this->rowPointers[ row + 1 ];
   while( elementPtr < rowEnd && this->columnIndexes[ elementPtr ] < column ) elementPtr++;
   if( elementPtr == rowEnd )
      return false;
   if( this->columnIndexes[ elementPtr ] == column )
   {
      this->values[ elementPtr ] = thisElementMultiplicator * this->values[ elementPtr ] + value;
      return true;
   }
   else
      if( this->columnIndexes[ elementPtr ] == this->columns )
      {
         this->columnIndexes[ elementPtr ] = column;
         this->values[ elementPtr ] = value;
         return true;
      }
      else
      {
         IndexType j = rowEnd - 1;
         while( j > elementPtr )
         {
            this->columnIndexes[ j ] = this->columnIndexes[ j - 1 ];
            this->values[ j ] = this->values[ j - 1 ];
            j--;
         }
         this->columnIndexes[ elementPtr ] = column;
         this->values[ elementPtr ] = value;
         return true;
      }
   return false;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlCSRMatrix< Real, Device, Index >::addElement( const IndexType row,
                                                      const IndexType column,
                                                      const RealType& value,
                                                      const RealType& thisElementMultiplicator )
{
   return this->addElementFast( row, column, value, thisElementMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlCSRMatrix< Real, Device, Index > :: setRowFast( const IndexType row,
                                                        const IndexType* columnIndexes,
                                                        const RealType* values,
                                                        const IndexType elements )
{
   IndexType elementPointer = this->rowPointers[ row ];
   const IndexType rowLength = this->rowPointers[ row + 1 ] - elementPointer;
   if( elements > rowLength )
      return false;

   for( IndexType i = 0; i < elements; i++ )
   {
      this->columnIndexes[ elementPointer ] = columnIndexes[ i ];
      this->values[ elementPointer ] = values[ i ];
      elementPointer++;
   }
   for( IndexType i = elements; i < rowLength; i++ )
      this->columnIndexes[ elementPointer++ ] = this->getColumns();
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlCSRMatrix< Real, Device, Index > :: setRow( const IndexType row,
                                                    const IndexType* columnIndexes,
                                                    const RealType* values,
                                                    const IndexType elements )
{
   return this->setRowFast( row, columnIndexes, values, elements );
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool tnlCSRMatrix< Real, Device, Index > :: addRowFast( const IndexType row,
                                                        const IndexType* columns,
                                                        const RealType* values,
                                                        const IndexType numberOfElements,
                                                        const RealType& thisElementMultiplicator )
{
   // TODO: implement
   return false;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlCSRMatrix< Real, Device, Index > :: addRow( const IndexType row,
                                                    const IndexType* columns,
                                                    const RealType* values,
                                                    const IndexType numberOfElements,
                                                    const RealType& thisElementMultiplicator )
{
   return this->addRowFast( row, columns, values, numberOfElements, thisElementMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Real tnlCSRMatrix< Real, Device, Index >::getElementFast( const IndexType row,
                                                          const IndexType column ) const
{
   IndexType elementPtr = this->rowPointers[ row ];
   const IndexType rowEnd = this->rowPointers[ row + 1 ];
   while( elementPtr < rowEnd && this->columnIndexes[ elementPtr ] < column )
      elementPtr++;
   if( elementPtr < rowEnd && this->columnIndexes[ elementPtr ] == column )
      return this->values[ elementPtr ];
   return 0.0;
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlCSRMatrix< Real, Device, Index >::getElement( const IndexType row,
                                                      const IndexType column ) const
{
   return this->getElementFast( row, column );
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
void tnlCSRMatrix< Real, Device, Index >::getRowFast( const IndexType row,
                                                      IndexType* columns,
                                                      RealType* values ) const
{
   IndexType elementPointer = this->rowPointers[ row ];
   const IndexType rowLength = this->rowPointers[ row + 1 ] - elementPointer;
   for( IndexType i = 0; i < rowLength; i++ )
   {
      columns[ i ] = this->columnIndexes[ elementPointer ];
      values[ i ] = this->values[ elementPointer ];
      elementPointer++;
   }
}

template< typename Real,
          typename Device,
          typename Index >
void tnlCSRMatrix< Real, Device, Index >::getRow( const IndexType row,
                                                  IndexType* columns,
                                                  RealType* values ) const
{
   return this->getRowFast( row, columns, values );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
typename Vector::RealType tnlCSRMatrix< Real, Device, Index >::rowVectorProduct( const IndexType row,
                                                                                 const Vector& vector ) const
{
   Real result = 0.0;
   IndexType elementPtr = this->rowPointers[ row ];
   const IndexType rowEnd = this->rowPointers[ row + 1 ];
   while( elementPtr < rowEnd && this->columnIndexes[ elementPtr ] < this->columns )
   {
      const Index column = this->columnIndexes[ elementPtr ];
      result += this->values[ elementPtr++ ] * vector[ column ];
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename InVector, typename OutVector >
void tnlCSRMatrix< Real, Device, Index >::vectorProduct( const InVector& inVector,
                                                         OutVector& outVector ) const
{
   for( Index row = 0; row < this->getRows(); row ++ )
      outVector.setElement( row, rowVectorProduct( row, inVector ) );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Index2 >
void tnlCSRMatrix< Real, Device, Index >::addMatrix( const tnlCSRMatrix< Real2, Device, Index2 >& matrix,
                                                                          const RealType& matrixMultiplicator,
                                                                          const RealType& thisMatrixMultiplicator )
{
   tnlAssert( false, cerr << "TODO: implement" );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Index2 >
void tnlCSRMatrix< Real, Device, Index >::getTransposition( const tnlCSRMatrix< Real2, Device, Index2 >& matrix,
                                                                      const RealType& matrixMultiplicator )
{
   tnlAssert( false, cerr << "TODO: implement" );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
bool tnlCSRMatrix< Real, Device, Index >::performSORIteration( const Vector& b,
                                                                                    const IndexType row,
                                                                                    Vector& x,
                                                                                    const RealType& omega ) const
{
   tnlAssert( row >=0 && row < this->getRows(),
              cerr << "row = " << row
                   << " this->getRows() = " << this->getRows()
                   << " this->getName() = " << this->getName() << endl );

   RealType diagonalValue( 0.0 );
   RealType sum( 0.0 );

   IndexType elementPtr = this->rowPointers[ row ];
   const IndexType rowEnd = this->rowPointers[ row + 1 ];
   IndexType column;
   while( elementPtr < rowEnd && ( column = this->columnIndexes[ elementPtr ] ) < this->columns )
   {
      if( column == row )
         diagonalValue = this->values.getElement( elementPtr );
      else
         sum += this->values.getElement( row * this->diagonalsShift.getSize() + elementPtr ) * x. getElement( column );
      elementPtr++;
   }
   if( diagonalValue == ( Real ) 0.0 )
   {
      cerr << "There is zero on the diagonal in " << row << "-th row of thge matrix " << this->getName() << ". I cannot perform SOR iteration." << endl;
      return false;
   }
   x. setElement( row, x[ row ] + omega / diagonalValue * ( b[ row ] - sum ) );
   return true;
}


template< typename Real,
          typename Device,
          typename Index >
bool tnlCSRMatrix< Real, Device, Index >::save( tnlFile& file ) const
{
   if( ! tnlSparseMatrix< Real, Device, Index >::save( file ) ||
       ! this->rowPointers.save( file ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlCSRMatrix< Real, Device, Index >::load( tnlFile& file )
{
   if( ! tnlSparseMatrix< Real, Device, Index >::load( file ) ||
       ! this->rowPointers.load( file ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlCSRMatrix< Real, Device, Index >::save( const tnlString& fileName ) const
{
   return tnlObject::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlCSRMatrix< Real, Device, Index >::load( const tnlString& fileName )
{
   return tnlObject::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
void tnlCSRMatrix< Real, Device, Index >::print( ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";
      IndexType elementPtr = this->rowPointers[ row ];
      const IndexType rowEnd = this->rowPointers[ row + 1 ];
      while( elementPtr < rowEnd && this->columnIndexes[ elementPtr ] < this->columns )
      {
         const Index column = this->columnIndexes.getElement( elementPtr );
         str << " Col:" << column << "->" << this->values.getElement( elementPtr ) << "\t";
         elementPtr++;
      }
      str << endl;
   }
}

#endif /* TNLCSRMATRIX_IMPL_H_ */
