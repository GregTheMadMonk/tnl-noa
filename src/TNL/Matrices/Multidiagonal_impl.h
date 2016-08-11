/***************************************************************************
                          Multidiagonal.h  -  description
                             -------------------
    begin                : Dec 4, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/Multidiagonal.h>
#include <TNL/Containers/Vector.h>
#include <TNL/core/mfuncs.h>

namespace TNL {
namespace Matrices {   

template< typename Device >
class MultidiagonalDeviceDependentCode;

template< typename Real,
          typename Device,
          typename Index >
Multidiagonal< Real, Device, Index > :: Multidiagonal()
{
};

template< typename Real,
          typename Device,
          typename Index >
String Multidiagonal< Real, Device, Index > :: getType()
{
   return String( "Matrices::Multidiagonal< ") +
          String( TNL::getType< Real >() ) +
          String( ", " ) +
          Device :: getDeviceType() +
          String( " >" );
}

template< typename Real,
          typename Device,
          typename Index >
String Multidiagonal< Real, Device, Index >::getTypeVirtual() const
{
   return this->getType();
}

template< typename Real,
          typename Device,
          typename Index >
bool Multidiagonal< Real, Device, Index >::setDimensions( const IndexType rows,
                                                                   const IndexType columns )
{
   Assert( rows > 0 && columns > 0,
              std::cerr << "rows = " << rows
                   << " columns = " << columns << std::endl );
   if( ! Matrix< Real, Device, Index >::setDimensions( rows, columns ) )
      return false;
   if( this->diagonalsShift.getSize() != 0 )
   {
      if( ! this->values.setSize( min( this->rows, this->columns ) * this->diagonalsShift.getSize() ) )
         return false;
      this->values.setValue( 0.0 );
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool Multidiagonal< Real, Device, Index >::setCompressedRowsLengths( const CompressedRowsLengthsVector& rowLengths )
{
   /****
    * TODO: implement some check here similar to the one in the tridiagonal matrix
    */
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
Index Multidiagonal< Real, Device, Index >::getRowLength( const IndexType row ) const
{
   IndexType rowLength( 0 );
   for( IndexType i = 0; i < diagonalsShift.getSize(); i++ )
   {
      const IndexType column = row + diagonalsShift.getElement( i );
      if( column >= 0 && column < this->getColumns() )
         rowLength++;
   }
   return rowLength;
}

template< typename Real,
          typename Device,
          typename Index >
Index
Multidiagonal< Real, Device, Index >::
getMaxRowLength() const
{
   return diagonalsShift.getSize();
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
bool Multidiagonal< Real, Device, Index > :: setDiagonals(  const Vector& diagonals )
{
   Assert( diagonals.getSize() > 0,
              std::cerr << "New number of diagonals = " << diagonals.getSize() << std::endl );
   this->diagonalsShift.setLike( diagonals );
   this->diagonalsShift = diagonals;
   if( this->rows != 0 && this->columns != 0 )
   {
      if( ! this->values.setSize( min( this->rows, this->columns ) * this->diagonalsShift.getSize() ) )
         return false;
      this->values.setValue( 0.0 );
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
const Containers::Vector< Index, Device, Index >& Multidiagonal< Real, Device, Index > :: getDiagonals() const
{
   return this->diagonalsShift;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool Multidiagonal< Real, Device, Index > :: setLike( const Multidiagonal< Real2, Device2, Index2 >& matrix )
{
   if( ! this->setDimensions( matrix.getRows(), matrix.getColumns() ) )
      return false;
   if( ! setDiagonals( matrix.getDiagonals() ) )
      return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
Index Multidiagonal< Real, Device, Index > :: getNumberOfMatrixElements() const
{
   return this->values.getSize();
}

template< typename Real,
          typename Device,
          typename Index >
Index Multidiagonal< Real, Device, Index > :: getNumberOfNonzeroMatrixElements() const
{
   IndexType nonzeroElements;
   for( IndexType i = 0; i < this->values.getSize(); i++ )
      if( this->values.getElement( i ) != 0 )
         nonzeroElements++;
   return nonzeroElements;
}

template< typename Real,
          typename Device,
          typename Index >
void Multidiagonal< Real, Device, Index > :: reset()
{
   this->rows = 0;
   this->columns = 0;
   this->values.reset();
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool Multidiagonal< Real, Device, Index >::operator == ( const Multidiagonal< Real2, Device2, Index2 >& matrix ) const
{
   Assert( this->getRows() == matrix.getRows() &&
              this->getColumns() == matrix.getColumns(),
              std::cerr << "this->getRows() = " << this->getRows()
                   << " matrix.getRows() = " << matrix.getRows()
                   << " this->getColumns() = " << this->getColumns()
                   << " matrix.getColumns() = " << matrix.getColumns() );
   return ( this->diagonals == matrix.diagonals &&
            this->values == matrix.values );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool Multidiagonal< Real, Device, Index >::operator != ( const Multidiagonal< Real2, Device2, Index2 >& matrix ) const
{
   return ! ( ( *this ) == matrix );
}

template< typename Real,
          typename Device,
          typename Index >
void Multidiagonal< Real, Device, Index >::setValue( const RealType& v )
{
   this->values.setValue( v );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool Multidiagonal< Real, Device, Index > :: setElementFast( const IndexType row,
                                                                      const IndexType column,
                                                                      const Real& value )
{
   IndexType index;
   if( ! this->getElementIndexFast( row, column, index  ) )
      return false;
   this->values[ index ] = value;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool Multidiagonal< Real, Device, Index > :: setElement( const IndexType row,
                                                                  const IndexType column,
                                                                  const Real& value )
{
   IndexType index;
   if( ! this->getElementIndex( row, column, index  ) )
      return false;
   this->values.setElement( index, value );
   return true;
}


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool Multidiagonal< Real, Device, Index > :: addElementFast( const IndexType row,
                                                                      const IndexType column,
                                                                      const RealType& value,
                                                                      const RealType& thisElementMultiplicator )
{
   Index index;
   if( ! this->getElementIndexFast( row, column, index  ) )
      return false;
   RealType& aux = this->values[ index ];
   aux = thisElementMultiplicator * aux + value;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool Multidiagonal< Real, Device, Index > :: addElement( const IndexType row,
                                                                  const IndexType column,
                                                                  const RealType& value,
                                                                  const RealType& thisElementMultiplicator )
{
   Index index;
   if( ! this->getElementIndex( row, column, index  ) )
      return false;
   this->values.setElement( index, thisElementMultiplicator * this->values.getElement( index ) + value );
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool Multidiagonal< Real, Device, Index > :: setRowFast( const IndexType row,
                                                                  const IndexType* columns,
                                                                  const RealType* values,
                                                                  const IndexType numberOfElements )
{
   return this->addRowFast( row, columns, values, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index >
bool Multidiagonal< Real, Device, Index > :: setRow( const IndexType row,
                                                              const Index* columns,
                                                              const Real* values,
                                                              const Index numberOfElements )
{
   return this->addRow( row, columns, values, numberOfElements, 0.0 );
}


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool Multidiagonal< Real, Device, Index > :: addRowFast( const IndexType row,
                                                                  const IndexType* columns,
                                                                  const RealType* values,
                                                                  const IndexType numberOfElements,
                                                                  const RealType& thisElementMultiplicator )
{
   if( this->diagonalsShift.getSize() < numberOfElements )
      return false;
   typedef MultidiagonalDeviceDependentCode< Device > DDCType;
   const IndexType elements = min( this->diagonalsShift.getSize(), numberOfElements );
   IndexType i( 0 );
   while( i < elements )
   {
      const IndexType index = DDCType::getElementIndex( this->getRows(), this->diagonalsShift.getSize(), row, i );
      RealType& aux = this->values[ index ];
      aux = thisElementMultiplicator * aux + values[ i ];
      i++;
   }
   while( i < this->diagonalsShift.getSize() )
   {
      const IndexType index = DDCType::getElementIndex( this->getRows(), this->diagonalsShift.getSize(), row, i );
      this->values[ index ] = 0;
      i++;
   }
   return true;

}

template< typename Real,
          typename Device,
          typename Index >
bool Multidiagonal< Real, Device, Index > :: addRow( const IndexType row,
                                                              const Index* columns,
                                                              const Real* values,
                                                              const Index numberOfElements,
                                                              const RealType& thisElementMultiplicator )
{
   if( this->diagonalsShift.getSize() < numberOfElements )
      return false;
   typedef MultidiagonalDeviceDependentCode< Device > DDCType;
   const IndexType elements = min( this->diagonalsShift.getSize(), numberOfElements );
   IndexType i( 0 );
   while( i < elements )
   {
      const IndexType index = DDCType::getElementIndex( this->getRows(), this->diagonalsShift.getSize(), row, i );
      if( thisElementMultiplicator == 0.0 )
         this->values.setElement( index, values[ i ] );
      else
         this->values.setElement( index, thisElementMultiplicator * this->values.getElement( index ) + values[ i ] );
      i++;
   }
   while( i < this->diagonalsShift.getSize() )
   {
      const IndexType index = DDCType::getElementIndex( this->getRows(), this->diagonalsShift.getSize(), row, i );
      this->values.setElement( index, 0 );
      i++;
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Real Multidiagonal< Real, Device, Index >::getElementFast( const IndexType row,
                                                                    const IndexType column ) const
{
   Index index;
   if( ! this->getElementIndexFast( row, column, index  ) )
      return 0.0;
   return this->values[ index ];
}

template< typename Real,
          typename Device,
          typename Index >
Real Multidiagonal< Real, Device, Index >::getElement( const IndexType row,
                                                                const IndexType column ) const
{
   Index index;
   if( ! this->getElementIndex( row, column, index  ) )
      return 0.0;
   return this->values.getElement( index );
}


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
void Multidiagonal< Real, Device, Index >::getRowFast( const IndexType row,
                                                                IndexType* columns,
                                                                RealType* values ) const
{
   IndexType pointer( 0 );
   for( IndexType i = 0; i < diagonalsShift.getSize(); i++ )
   {
      const IndexType column = row + diagonalsShift[ i ];
      if( column >= 0 && column < this->getColumns() )
      {
         columns[ pointer ] = column;
         values[ pointer ] = this->getElementFast( row, column );
         pointer++;
      }
   }
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
typename Multidiagonal< Real, Device, Index >::MatrixRow
Multidiagonal< Real, Device, Index >::
getRow( const IndexType rowIndex )
{
   IndexType firstRowElement( 0 );
   while( rowIndex + this->diagonalsShift[ firstRowElement ] < 0 )
      firstRowElement ++;

   IndexType firstRowElementIndex;
   this->getElementIndexFast( rowIndex, rowIndex + this->diagonalsShift[ firstRowElement ], firstRowElementIndex );
   if( std::is_same< Device, Devices::Host >::value )
      return MatrixRow( &this->values.getData()[ firstRowElementIndex ],
                        &this->diagonalsShift.getData()[ firstRowElement ],
                        this->diagonalsShift.getSize() - firstRowElement,
                        rowIndex,
                        this->getColumns(),
                        1 );
   if( std::is_same< Device, Devices::Cuda >::value )
      return MatrixRow( &this->values.getData()[ firstRowElementIndex ],
                        &this->diagonalsShift.getData()[ firstRowElement ],
                        this->diagonalsShift.getSize()- firstRowElement,
                        rowIndex,
                        this->getColumns(),
                        this->rows );

}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const typename Multidiagonal< Real, Device, Index >::MatrixRow
Multidiagonal< Real, Device, Index >::
getRow( const IndexType rowIndex ) const
{
   IndexType firstRowElement( 0 );
   while( rowIndex + this->diagonalsShift[ firstRowElement ] < 0 )
      firstRowElement ++;

   IndexType firstRowElementIndex;
   this->getElementIndexFast( rowIndex, rowIndex + this->diagonalsShift[ firstRowElement ], firstRowElementIndex );
   if( std::is_same< Device, Devices::Host >::value )
      return MatrixRow( &this->values.getData()[ firstRowElementIndex ],
                        &this->diagonalsShift.getData()[ firstRowElement ],
                        this->diagonalsShift.getSize() - firstRowElement,
                        rowIndex,
                        this->getColumns(),
                        1 );
   if( std::is_same< Device, Devices::Cuda >::value )
      return MatrixRow( &this->values.getData()[ firstRowElementIndex ],
                        &this->diagonalsShift.getData()[ firstRowElement ],
                        this->diagonalsShift.getSize()- firstRowElement,
                        rowIndex,
                        this->getColumns(),
                        this->rows );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
__cuda_callable__
typename Vector::RealType Multidiagonal< Real, Device, Index >::rowVectorProduct( const IndexType row,
                                                                                           const Vector& vector ) const
{
   typedef MultidiagonalDeviceDependentCode< Device > DDCType;
   Real result = 0.0;
   for( Index i = 0;
        i < this->diagonalsShift.getSize();
        i ++ )
   {
      const Index column = row + this->diagonalsShift[ i ];
      if( column >= 0 && column < this->getColumns() )
         result += this->values[
                      DDCType::getElementIndex( this->getRows(),
                                                this->diagonalsShift.getSize(),
                                                row,
                                                i ) ] * vector[ column ];
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename InVector,
             typename OutVector >
void Multidiagonal< Real, Device, Index >::vectorProduct( const InVector& inVector,
                                                                   OutVector& outVector ) const
{
   Assert( this->getColumns() == inVector.getSize(),
            std::cerr << "Matrix columns: " << this->getColumns() << std::endl
                 << "Vector size: " << inVector.getSize() << std::endl );
   Assert( this->getRows() == outVector.getSize(),
               std::cerr << "Matrix rows: " << this->getRows() << std::endl
                    << "Vector size: " << outVector.getSize() << std::endl );

   DeviceDependentCode::vectorProduct( *this, inVector, outVector );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Index2 >
void Multidiagonal< Real, Device, Index > :: addMatrix( const Multidiagonal< Real2, Device, Index2 >& matrix,
                                                                 const RealType& matrixMultiplicator,
                                                                 const RealType& thisMatrixMultiplicator )
{
   Assert( false, std::cerr << "TODO: implement" );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Index2 >
void Multidiagonal< Real, Device, Index >::getTransposition( const Multidiagonal< Real2, Device, Index2 >& matrix,
                                                                      const RealType& matrixMultiplicator )
{
   Containers::Vector< Index > auxDiagonals;
   auxDiagonals.setLike( matrix.getDiagonals() );
   const Index numberOfDiagonals = matrix.getDiagonals().getSize();
   for( Index i = 0; i < numberOfDiagonals; i++ )
      auxDiagonals[ i ] = -1.0 * matrix.getDiagonals().getElement( numberOfDiagonals - i - 1 );
   this->setDimensions( matrix.getColumns(),
                        matrix.getRows() );
   this->setDiagonals( auxDiagonals );
   for( Index row = 0; row < matrix.getRows(); row++ )
      for( Index diagonal = 0; diagonal < numberOfDiagonals; diagonal++ )
      {
         const Index column = row + matrix.getDiagonals().getElement( diagonal );
         if( column >= 0 && column < matrix.getColumns() )
            this->setElement( column, row, matrixMultiplicator * matrix.getElement( row, column ) );
      }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
bool Multidiagonal< Real, Device, Index > :: performSORIteration( const Vector& b,
                                                                           const IndexType row,
                                                                           Vector& x,
                                                                           const RealType& omega ) const
{
   Assert( row >=0 && row < this->getRows(),
              std::cerr << "row = " << row
                   << " this->getRows() = " << this->getRows() << std::endl );

   RealType diagonalValue( 0.0 );
   RealType sum( 0.0 );

   const IndexType maxRowLength = this->getMaxRowLength();

   for( IndexType i = 0; i < maxRowLength; i++ )
   {
      const IndexType column = row + this->diagonalsShift[ i ];
      if( column >= 0 && column < this->getColumns() )
      {
         IndexType elementIndex;
         this->getElementIndex( row, column, elementIndex );
         if( column == row )
            diagonalValue = this->values[ elementIndex ];
         else
            sum += this->values[ elementIndex ] * x[ column ];
      }
   }
   if( diagonalValue == ( Real ) 0.0 )
   {
      std::cerr << "There is zero on the diagonal in " << row << "-th row of thge matrix. I cannot perform SOR iteration." << std::endl;
      return false;
   }
   x[ row ] = ( 1.0 - omega ) * x[ row ] + omega / diagonalValue * ( b[ row ] - sum );
   return true;
}


template< typename Real,
          typename Device,
          typename Index >
bool Multidiagonal< Real, Device, Index >::save( File& file ) const
{
   if( ! Matrix< Real, Device, Index >::save( file ) ) return false;
   if( ! this->values.save( file ) ) return false;
   if( ! this->diagonalsShift.save( file ) ) return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool Multidiagonal< Real, Device, Index >::load( File& file )
{
   if( ! Matrix< Real, Device, Index >::load( file ) ) return false;
   if( ! this->values.load( file ) ) return false;
   if( ! this->diagonalsShift.load( file ) ) return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool Multidiagonal< Real, Device, Index >::save( const String& fileName ) const
{
   return Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
bool Multidiagonal< Real, Device, Index >::load( const String& fileName )
{
   return Object::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
void Multidiagonal< Real, Device, Index >::print( std::ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";
      for( IndexType i = 0; i < this->diagonalsShift.getSize(); i++ )
      {
         const IndexType column = row + diagonalsShift.getElement( i );
         if( column >=0 && column < this->columns )
            str << " Col:" << column << "->" << this->getElement( row, column ) << "\t";
      }
      str << std::endl;
   }
}

template< typename Real,
          typename Device,
          typename Index >
bool Multidiagonal< Real, Device, Index >::getElementIndex( const IndexType row,
                                                                     const IndexType column,
                                                                     Index& index ) const
{
   Assert( row >=0 && row < this->rows,
            std::cerr << "row = " << row
                 << " this->rows = " << this->rows << std::endl );
   Assert( column >=0 && column < this->columns,
            std::cerr << "column = " << column
                 << " this->columns = " << this->columns << std::endl );

   typedef MultidiagonalDeviceDependentCode< Device > DDCType;
   IndexType i( 0 );
   while( i < this->diagonalsShift.getSize() )
   {
      if( diagonalsShift.getElement( i ) == column - row )
      {
         index = DDCType::getElementIndex( this->getRows(), this->diagonalsShift.getSize(), row, i );
         return true;
      }
      i++;
   }
   return false;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool Multidiagonal< Real, Device, Index >::getElementIndexFast( const IndexType row,
                                                                         const IndexType column,
                                                                         Index& index ) const
{
   Assert( row >=0 && row < this->rows,
            std::cerr << "row = " << row
                 << " this->rows = " << this->rows << std::endl );
   Assert( column >=0 && column < this->columns,
            std::cerr << "column = " << column
                 << " this->columns = " << this->columns << std::endl );

   typedef MultidiagonalDeviceDependentCode< Device > DDCType;
   IndexType i( 0 );
   while( i < this->diagonalsShift.getSize() )
   {
      if( diagonalsShift[ i ] == column - row )
      {
         index = DDCType::getElementIndex( this->getRows(), this->diagonalsShift.getSize(), row, i );
         return true;
      }
      i++;
   }
   return false;
}

template<>
class MultidiagonalDeviceDependentCode< Devices::Host >
{
   public:

      typedef Devices::Host Device;

      template< typename Index >
      __cuda_callable__
      static Index getElementIndex( const Index rows,
                                    const Index diagonals,
                                    const Index row,
                                    const Index diagonal )
      {
         return row*diagonals + diagonal;
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const Multidiagonal< Real, Device, Index >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
#ifdef HAVE_OPENMP
#pragma omp parallel for if( Devices::Host::isOMPEnabled() )
#endif
         for( Index row = 0; row < matrix.getRows(); row ++ )
            outVector[ row ] = matrix.rowVectorProduct( row, inVector );
      }
};

template<>
class MultidiagonalDeviceDependentCode< Devices::Cuda >
{
   public:

      typedef Devices::Cuda Device;

      template< typename Index >
      __cuda_callable__
      static Index getElementIndex( const Index rows,
                                    const Index diagonals,
                                    const Index row,
                                    const Index diagonal )
      {
         return diagonal*rows + row;
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const Multidiagonal< Real, Device, Index >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         MatrixVectorProductCuda( matrix, inVector, outVector );
      }
};

} // namespace Matrices
} // namespace TNL
