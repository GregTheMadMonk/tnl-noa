/***************************************************************************
                          Tridiagonal_impl.h  -  description
                             -------------------
    begin                : Nov 30, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Assert.h>
#include <TNL/Matrices/Tridiagonal.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
namespace Matrices {   

template< typename Device >
class TridiagonalDeviceDependentCode;

template< typename Real,
          typename Device,
          typename Index >
Tridiagonal< Real, Device, Index >::Tridiagonal()
{
}

template< typename Real,
          typename Device,
          typename Index >
String Tridiagonal< Real, Device, Index >::getType()
{
   return String( "Matrices::Tridiagonal< " ) +
          String( TNL::getType< RealType >() ) + ", " +
          String( Device :: getDeviceType() ) + ", " +
          String( TNL::getType< IndexType >() ) + " >";
}

template< typename Real,
          typename Device,
          typename Index >
String Tridiagonal< Real, Device, Index >::getTypeVirtual() const
{
   return this->getType();
}

template< typename Real,
          typename Device,
          typename Index >
String Tridiagonal< Real, Device, Index >::getSerializationType()
{
   return getType();
}

template< typename Real,
          typename Device,
          typename Index >
String Tridiagonal< Real, Device, Index >::getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index >
void Tridiagonal< Real, Device, Index >::setDimensions( const IndexType rows,
                                                        const IndexType columns )
{
   Matrix< Real, Device, Index >::setDimensions( rows, columns );
   values.setSize( 3*min( rows, columns ) );
   this->values.setValue( 0.0 );
}

template< typename Real,
          typename Device,
          typename Index >
void Tridiagonal< Real, Device, Index >::setCompressedRowLengths( ConstCompressedRowLengthsVectorView rowLengths )
{
   if( rowLengths[ 0 ] > 2 )
      throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
   const IndexType diagonalLength = min( this->getRows(), this->getColumns() );
   for( Index i = 1; i < diagonalLength-1; i++ )
      if( rowLengths[ i ] > 3 )
         throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
   if( this->getRows() > this->getColumns() )
      if( rowLengths[ this->getRows()-1 ] > 1 )
         throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
   if( this->getRows() == this->getColumns() )
      if( rowLengths[ this->getRows()-1 ] > 2 )
         throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
   if( this->getRows() < this->getColumns() )
      if( rowLengths[ this->getRows()-1 ] > 3 )
         throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
}

template< typename Real,
          typename Device,
          typename Index >
Index Tridiagonal< Real, Device, Index >::getRowLength( const IndexType row ) const
{
   return this->getRowLengthFast( row );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index Tridiagonal< Real, Device, Index >::getRowLengthFast( const IndexType row ) const
{
   const IndexType diagonalLength = min( this->getRows(), this->getColumns() );
   if( row == 0 )
      return 2;
   if( row > 0 && row < diagonalLength - 1 )
      return 3;
   if( this->getRows() > this->getColumns() )
      return 1;
   if( this->getRows() == this->getColumns() )
      return 2;
   return 3;
}

template< typename Real,
          typename Device,
          typename Index >
Index Tridiagonal< Real, Device, Index >::getMaxRowLength() const
{
   return 3;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2, typename Device2, typename Index2 >
void Tridiagonal< Real, Device, Index >::setLike( const Tridiagonal< Real2, Device2, Index2 >& m )
{
   this->setDimensions( m.getRows(), m.getColumns() );
}

template< typename Real,
          typename Device,
          typename Index >
Index Tridiagonal< Real, Device, Index >::getNumberOfMatrixElements() const
{
   return 3 * min( this->getRows(), this->getColumns() );
}

template< typename Real,
          typename Device,
          typename Index >
Index Tridiagonal< Real, Device, Index > :: getNumberOfNonzeroMatrixElements() const
{
   IndexType nonzeroElements = 0;
   for( IndexType i = 0; i < this->values.getSize(); i++ )
      if( this->values.getElement( i ) != 0 )
         nonzeroElements++;
   return nonzeroElements;
}

template< typename Real,
          typename Device,
          typename Index >
Index
Tridiagonal< Real, Device, Index >::
getMaxRowlength() const
{
   return 3;
}

template< typename Real,
          typename Device,
          typename Index >
void Tridiagonal< Real, Device, Index >::reset()
{
   Matrix< Real, Device, Index >::reset();
   this->values.reset();
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2, typename Device2, typename Index2 >
bool Tridiagonal< Real, Device, Index >::operator == ( const Tridiagonal< Real2, Device2, Index2 >& matrix ) const
{
   return this->values == matrix.values;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2, typename Device2, typename Index2 >
bool Tridiagonal< Real, Device, Index >::operator != ( const Tridiagonal< Real2, Device2, Index2 >& matrix ) const
{
   return this->values != matrix.values;
}

template< typename Real,
          typename Device,
          typename Index >
void Tridiagonal< Real, Device, Index >::setValue( const RealType& v )
{
   this->values.setValue( v );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool Tridiagonal< Real, Device, Index >::setElementFast( const IndexType row,
                                                                  const IndexType column,
                                                                  const RealType& value )
{
   this->values[ this->getElementIndex( row, column ) ] = value;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool Tridiagonal< Real, Device, Index >::setElement( const IndexType row,
                                                              const IndexType column,
                                                              const RealType& value )
{
   this->values.setElement( this->getElementIndex( row, column ), value );
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool Tridiagonal< Real, Device, Index >::addElementFast( const IndexType row,
                                                                  const IndexType column,
                                                                  const RealType& value,
                                                                  const RealType& thisElementMultiplicator )
{
   const Index i = this->getElementIndex( row, column );
   this->values[ i ] = thisElementMultiplicator*this->values[ i ] + value;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool Tridiagonal< Real, Device, Index >::addElement( const IndexType row,
                                                              const IndexType column,
                                                              const RealType& value,
                                                              const RealType& thisElementMultiplicator )
{
   const Index i = this->getElementIndex( row, column );
   this->values.setElement( i, thisElementMultiplicator * this->values.getElement( i ) + value );
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool Tridiagonal< Real, Device, Index >::setRowFast( const IndexType row,
                                                              const IndexType* columns,
                                                              const RealType* values,
                                                              const IndexType elements )
{
   TNL_ASSERT( elements <= this->columns,
            std::cerr << " elements = " << elements
                 << " this->columns = " << this->columns );
   return this->addRowFast( row, columns, values, elements, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index >
bool Tridiagonal< Real, Device, Index >::setRow( const IndexType row,
                                                          const IndexType* columns,
                                                          const RealType* values,
                                                          const IndexType elements )
{
   TNL_ASSERT( elements <= this->columns,
            std::cerr << " elements = " << elements
                 << " this->columns = " << this->columns );
   return this->addRow( row, columns, values, elements, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool Tridiagonal< Real, Device, Index >::addRowFast( const IndexType row,
                                                              const IndexType* columns,
                                                              const RealType* values,
                                                              const IndexType elements,
                                                              const RealType& thisRowMultiplicator )
{
   TNL_ASSERT( elements <= this->columns,
            std::cerr << " elements = " << elements
                 << " this->columns = " << this->columns );
   if( elements > 3 )
      return false;
   for( IndexType i = 0; i < elements; i++ )
   {
      const IndexType& column = columns[ i ];
      if( column < row - 1 || column > row + 1 )
         return false;
      addElementFast( row, column, values[ i ], thisRowMultiplicator );
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool Tridiagonal< Real, Device, Index >::addRow( const IndexType row,
                                                          const IndexType* columns,
                                                          const RealType* values,
                                                          const IndexType elements,
                                                          const RealType& thisRowMultiplicator )
{
   TNL_ASSERT( elements <= this->columns,
            std::cerr << " elements = " << elements
                 << " this->columns = " << this->columns );
   if( elements > 3 )
      return false;
   for( IndexType i = 0; i < elements; i++ )
   {
      const IndexType column = columns[ i ];
      if( column < row - 1 || column > row + 1 )
         return false;
      addElement( row, column, values[ i ], thisRowMultiplicator );
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Real Tridiagonal< Real, Device, Index >::getElementFast( const IndexType row,
                                                                  const IndexType column ) const
{
   if( abs( column - row ) > 1 )
      return 0.0;
   return this->values[ this->getElementIndex( row, column ) ];
}

template< typename Real,
          typename Device,
          typename Index >
Real Tridiagonal< Real, Device, Index >::getElement( const IndexType row,
                                                              const IndexType column ) const
{
   if( abs( column - row ) > 1 )
      return 0.0;
   return this->values.getElement( this->getElementIndex( row, column ) );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
void Tridiagonal< Real, Device, Index >::getRowFast( const IndexType row,
                                                              IndexType* columns,
                                                              RealType* values ) const
{
   IndexType elementPointer( 0 );
   for( IndexType i = -1; i <= 1; i++ )
   {
      const IndexType column = row + 1;
      if( column >= 0 && column < this->getColumns() )
      {
         columns[ elementPointer ] = column;
         values[ elementPointer ] = this->values[ this->getElementIndex( row, column ) ];
         elementPointer++;
      }
   }
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
typename Tridiagonal< Real, Device, Index >::MatrixRow
Tridiagonal< Real, Device, Index >::
getRow( const IndexType rowIndex )
{
   if( std::is_same< Device, Devices::Host >::value )
      return MatrixRow( &this->values.getData()[ this->getElementIndex( rowIndex, rowIndex ) ],
                        rowIndex,
                        this->getColumns(),
                        1 );
   if( std::is_same< Device, Devices::Cuda >::value )
      return MatrixRow( &this->values.getData()[ this->getElementIndex( rowIndex, rowIndex ) ],
                        rowIndex,
                        this->getColumns(),
                        this->rows );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const typename Tridiagonal< Real, Device, Index >::MatrixRow
Tridiagonal< Real, Device, Index >::
getRow( const IndexType rowIndex ) const
{
   throw Exceptions::NotImplementedError();
}


template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
__cuda_callable__
typename Vector::RealType Tridiagonal< Real, Device, Index >::rowVectorProduct( const IndexType row,
                                                                                         const Vector& vector ) const
{
   return TridiagonalDeviceDependentCode< Device >::
             rowVectorProduct( this->rows,
                               this->values,
                               row,
                               vector );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename InVector,
             typename OutVector >
void Tridiagonal< Real, Device, Index >::vectorProduct( const InVector& inVector,
                                                                 OutVector& outVector ) const
{
   TNL_ASSERT( this->getColumns() == inVector.getSize(),
            std::cerr << "Matrix columns: " << this->getColumns() << std::endl
                 << "Vector size: " << inVector.getSize() << std::endl );
   TNL_ASSERT( this->getRows() == outVector.getSize(),
               std::cerr << "Matrix rows: " << this->getRows() << std::endl
                    << "Vector size: " << outVector.getSize() << std::endl );

   DeviceDependentCode::vectorProduct( *this, inVector, outVector );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2, typename Index2 >
void Tridiagonal< Real, Device, Index >::addMatrix( const Tridiagonal< Real2, Device, Index2 >& matrix,
                                                    const RealType& matrixMultiplicator,
                                                    const RealType& thisMatrixMultiplicator )
{
   TNL_ASSERT( this->getRows() == matrix.getRows(),
            std::cerr << "This matrix columns: " << this->getColumns() << std::endl
                 << "This matrix rows: " << this->getRows() << std::endl );

   if( thisMatrixMultiplicator == 1.0 )
      this->values += matrixMultiplicator * matrix.values;
   else
      this->values = thisMatrixMultiplicator * this->values + matrixMultiplicator * matrix.values;
}

#ifdef HAVE_CUDA
template< typename Real,
          typename Real2,
          typename Index,
          typename Index2 >
__global__ void TridiagonalTranspositionCudaKernel( const Tridiagonal< Real2, Devices::Cuda, Index2 >* inMatrix,
                                                             Tridiagonal< Real, Devices::Cuda, Index >* outMatrix,
                                                             const Real matrixMultiplicator,
                                                             const Index gridIdx )
{
   const Index rowIdx = ( gridIdx * Devices::Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( rowIdx < inMatrix->getRows() )
   {
      if( rowIdx > 0 )
        outMatrix->setElementFast( rowIdx-1,
                                   rowIdx,
                                   matrixMultiplicator * inMatrix->getElementFast( rowIdx, rowIdx-1 ) );
      outMatrix->setElementFast( rowIdx,
                                 rowIdx,
                                 matrixMultiplicator * inMatrix->getElementFast( rowIdx, rowIdx ) );
      if( rowIdx < inMatrix->getRows()-1 )
         outMatrix->setElementFast( rowIdx+1,
                                    rowIdx,
                                    matrixMultiplicator * inMatrix->getElementFast( rowIdx, rowIdx+1 ) );
   }
}
#endif

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2, typename Index2 >
void Tridiagonal< Real, Device, Index >::getTransposition( const Tridiagonal< Real2, Device, Index2 >& matrix,
                                                                    const RealType& matrixMultiplicator )
{
   TNL_ASSERT( this->getRows() == matrix.getRows(),
               std::cerr << "This matrix rows: " << this->getRows() << std::endl
                    << "That matrix rows: " << matrix.getRows() << std::endl );
   if( std::is_same< Device, Devices::Host >::value )
   {
      const IndexType& rows = matrix.getRows();
      for( IndexType i = 1; i < rows; i++ )
      {
         RealType aux = matrix. getElement( i, i - 1 );
         this->setElement( i, i - 1, matrix.getElement( i - 1, i ) );
         this->setElement( i, i, matrix.getElement( i, i ) );
         this->setElement( i - 1, i, aux );
      }
   }
   if( std::is_same< Device, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      Tridiagonal* kernel_this = Devices::Cuda::passToDevice( *this );
      typedef  Tridiagonal< Real2, Device, Index2 > InMatrixType;
      InMatrixType* kernel_inMatrix = Devices::Cuda::passToDevice( matrix );
      dim3 cudaBlockSize( 256 ), cudaGridSize( Devices::Cuda::getMaxGridSize() );
      const IndexType cudaBlocks = roundUpDivision( matrix.getRows(), cudaBlockSize.x );
      const IndexType cudaGrids = roundUpDivision( cudaBlocks, Devices::Cuda::getMaxGridSize() );
      for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
      {
         if( gridIdx == cudaGrids - 1 )
            cudaGridSize.x = cudaBlocks % Devices::Cuda::getMaxGridSize();
         TridiagonalTranspositionCudaKernel<<< cudaGridSize, cudaBlockSize >>>
                                                    ( kernel_inMatrix,
                                                      kernel_this,
                                                      matrixMultiplicator,
                                                      gridIdx );
      }
      Devices::Cuda::freeFromDevice( kernel_this );
      Devices::Cuda::freeFromDevice( kernel_inMatrix );
      TNL_CHECK_CUDA_DEVICE;
#endif
   }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector1, typename Vector2 >
__cuda_callable__
void Tridiagonal< Real, Device, Index >::performSORIteration( const Vector1& b,
                                                              const IndexType row,
                                                              Vector2& x,
                                                              const RealType& omega ) const
{
   RealType sum( 0.0 );
   if( row > 0 )
      sum += this->getElementFast( row, row - 1 ) * x[ row - 1 ];
   if( row < this->getColumns() - 1 )
      sum += this->getElementFast( row, row + 1 ) * x[ row + 1 ];
   x[ row ] = ( 1.0 - omega ) * x[ row ] + omega / this->getElementFast( row, row ) * ( b[ row ] - sum );
}


// copy assignment
template< typename Real,
          typename Device,
          typename Index >
Tridiagonal< Real, Device, Index >&
Tridiagonal< Real, Device, Index >::operator=( const Tridiagonal& matrix )
{
   this->setLike( matrix );
   this->values = matrix.values;
   return *this;
}

// cross-device copy assignment
template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2, typename Device2, typename Index2, typename >
Tridiagonal< Real, Device, Index >&
Tridiagonal< Real, Device, Index >::operator=( const Tridiagonal< Real2, Device2, Index2 >& matrix )
{
   static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Cuda >::value,
                  "unknown device" );
   static_assert( std::is_same< Device2, Devices::Host >::value || std::is_same< Device2, Devices::Cuda >::value,
                  "unknown device" );

   this->setLike( matrix );

   throw Exceptions::NotImplementedError("Cross-device assignment for the Tridiagonal format is not implemented yet.");
}


template< typename Real,
          typename Device,
          typename Index >
void Tridiagonal< Real, Device, Index >::save( File& file ) const
{
   Matrix< Real, Device, Index >::save( file );
   file << this->values;
}

template< typename Real,
          typename Device,
          typename Index >
void Tridiagonal< Real, Device, Index >::load( File& file )
{
   Matrix< Real, Device, Index >::load( file );
   file >> this->values;
}

template< typename Real,
          typename Device,
          typename Index >
void Tridiagonal< Real, Device, Index >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
void Tridiagonal< Real, Device, Index >::load( const String& fileName )
{
   Object::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
void Tridiagonal< Real, Device, Index >::print( std::ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";
      for( IndexType column = row - 1; column < row + 2; column++ )
         if( column >= 0 && column < this->columns )
            str << " Col:" << column << "->" << this->getElement( row, column ) << "\t";
      str << std::endl;
   }
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index Tridiagonal< Real, Device, Index >::getElementIndex( const IndexType row,
                                                                    const IndexType column ) const
{
   TNL_ASSERT( row >= 0 && column >= 0 && row < this->rows && column < this->rows,
              std::cerr << " this->rows = " << this->rows
                   << " row = " << row << " column = " << column );
   TNL_ASSERT( abs( row - column ) < 2,
              std::cerr << "row = " << row << " column = " << column << std::endl );
   return TridiagonalDeviceDependentCode< Device >::getElementIndex( this->rows, row, column );
}

template<>
class TridiagonalDeviceDependentCode< Devices::Host >
{
   public:

      typedef Devices::Host Device;

      template< typename Index >
      __cuda_callable__
      static Index getElementIndex( const Index rows,
                                    const Index row,
                                    const Index column )
      {
         return 2*row + column;
      }

      template< typename Vector,
                typename Index,
                typename ValuesType  >
      __cuda_callable__
      static typename Vector::RealType rowVectorProduct( const Index rows,
                                                         const ValuesType& values,
                                                         const Index row,
                                                         const Vector& vector )
      {
         if( row == 0 )
            return vector[ 0 ] * values[ 0 ] +
                   vector[ 1 ] * values[ 1 ];
         Index i = 3 * row;
         if( row == rows - 1 )
            return vector[ row - 1 ] * values[ i - 1 ] +
                   vector[ row ] * values[ i ];
         return vector[ row - 1 ] * values[ i - 1 ] +
                vector[ row ] * values[ i ] +
                vector[ row + 1 ] * values[ i + 1 ];
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const Tridiagonal< Real, Device, Index >& matrix,
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
class TridiagonalDeviceDependentCode< Devices::Cuda >
{
   public:
 
      typedef Devices::Cuda Device;

      template< typename Index >
      __cuda_callable__
      static Index getElementIndex( const Index rows,
                                    const Index row,
                                    const Index column )
      {
         return ( column - row + 1 )*rows + row - 1;
      }

      template< typename Vector,
                typename Index,
                typename ValuesType >
      __cuda_callable__
      static typename Vector::RealType rowVectorProduct( const Index rows,
                                                         const ValuesType& values,
                                                         const Index row,
                                                         const Vector& vector )
      {
         if( row == 0 )
            return vector[ 0 ] * values[ 0 ] +
                   vector[ 1 ] * values[ rows - 1 ];
         Index i = row - 1;
         if( row == rows - 1 )
            return vector[ row - 1 ] * values[ i ] +
                   vector[ row ] * values[ i + rows ];
         return vector[ row - 1 ] * values[ i ] +
                vector[ row ] * values[ i + rows ] +
                vector[ row + 1 ] * values[ i + 2*rows ];
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const Tridiagonal< Real, Device, Index >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         MatrixVectorProductCuda( matrix, inVector, outVector );
      }
};

} // namespace Matrices
} // namespace TNL
