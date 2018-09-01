/***************************************************************************
                          EllpackSymmetricGraph_impl.h  -  description
                             -------------------
    begin                : Aug 30, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/EllpackSymmetricGraph.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Math.h>

namespace TNL {
namespace Matrices {

template< typename Real,
          typename Device,
          typename Index >
EllpackSymmetricGraph< Real, Device, Index > :: EllpackSymmetricGraph()
: rowLengths( 0 ), alignedRows( 0 ), rearranged( false )
{
};

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
  __device__ __host__
#endif
Index EllpackSymmetricGraph< Real, Device, Index >::getRowLengthsInt() const
{
    return this->rowLengths;
}

template< typename Real,
          typename Device,
          typename Index >
Index EllpackSymmetricGraph< Real, Device, Index >::getAlignedRows() const
{
    return this->alignedRows;
}

template< typename Real,
          typename Device,
          typename Index >
String EllpackSymmetricGraph< Real, Device, Index > :: getType()
{
   return String( "EllpackSymmetricGraph< ") +
          String( TNL::getType< Real >() ) +
          String( ", " ) +
          Device::getDeviceType() +
          String( " >" );
}

template< typename Real,
          typename Device,
          typename Index >
String EllpackSymmetricGraph< Real, Device, Index >::getTypeVirtual() const
{
   return this->getType();
}

template< typename Real,
          typename Device,
          typename Index >
void EllpackSymmetricGraph< Real, Device, Index >::setDimensions( const IndexType rows,
                                                                  const IndexType columns )
{
   TNL_ASSERT( rows > 0 && columns > 0,
              std::cerr << "rows = " << rows
                   << " columns = " << columns << std::endl );
   this->rows = rows;
   this->columns = columns;   
   if( std::is_same< DeviceType, Devices::Cuda >::value )
      this->alignedRows = roundToMultiple( columns, Devices::Cuda::getWarpSize() );
   else this->alignedRows = rows;
   if( this->rowLengths != 0 )
   allocateElements();
}

template< typename Real,
          typename Device,
          typename Index >
void EllpackSymmetricGraph< Real, Device, Index >::setCompressedRowLengths( const CompressedRowLengthsVector& rowLengths )
{
   TNL_ASSERT( this->getRows() > 0, );
   TNL_ASSERT( this->getColumns() > 0, );
   //TNL_ASSERT( this->rowLengths > 0,
   //          std::cerr << "this->rowLengths = " << this->rowLengths );
   this->rowLengths = this->maxRowLength = rowLengths.max();
   this->permutationArray.setSize( this->getRows() );
   for( IndexType i = 0; i < this->getRows(); i++ )
      this->permutationArray.setElement( i, i );
   allocateElements();
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
__device__ __host__
#endif
Index EllpackSymmetricGraph< Real, Device, Index >::getRowsOfColor( IndexType color ) const
{
   return this->colorPointers.getElement( color + 1 ) - this->colorPointers.getElement( color );
}

/*
template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
__device__ __host__
#endif
void EllpackSymmetricGraph< Real, Device, Index >::computeColorsVector( Containers::Vector< Index, Device, Index >& colorsVector )
{
    this->numberOfColors = 0;

    for( IndexType i = this->getRows() - 1; i >= 0; i-- )
    {
        // init color array
        Containers::Vector< Index, Device, Index > usedColors;
        usedColors.setSize( this->numberOfColors );
        for( IndexType j = 0; j < this->numberOfColors; j++ )
            usedColors.setElement( j, 0 );

        // find all colors used in given row

        // optimization:
        //     load the whole row in sparse format
        //     traverse it while don't hit the padding index or end of the row
        //     for each nonzero element write -> usedColors.setElement( colorsVector.getElement( column ), 1 )
        IndexType* columns = new IndexType[ this->getRowLength( i ) ];
        RealType* values = new RealType[ this->getRowLength( i ) ];
        this->getRow( i, columns, values );
        for( IndexType j = 0; j < this->getRowLength( i ); j++ )
        {
            // we are only interested in symmetric part of the matrix
            if( columns[ j ] < i + 1 )
                continue;

            // if we hit padding index, there is no reason to continue iterations
            if( columns[ j ] == this->getPaddingIndex() )
                break;

            usedColors.setElement( colorsVector.getElement( columns[ j ] ), 1 );
        }
        delete [] columns;
        delete [] values;


       //for( IndexType j = i + 1; j < this->getColumns(); j++ )
       //     if( this->getElement( i, j ) != 0.0 )
       //         usedColors.setElement( colorsVector.getElement( j ), 1 );

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
*/

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
__device__ __host__
#endif
void EllpackSymmetricGraph< Real, Device, Index >::computePermutationArray()
{
   // init vector of colors and permutation array
   Containers::Vector< Index, Device, Index > colorsVector;
   colorsVector.setSize( this->getRows() );
   for( IndexType i = 0; i < this->getRows(); i++ )
   {
      colorsVector.setElement( i, 0 );
   }

   // compute colors for each row
   Matrix< Real, Device, Index >::computeColorsVector( colorsVector );

   // init color pointers
   this->colorPointers.setSize( this->getNumberOfColors() + 1 );

   // compute permutation
   IndexType position = 0;
   for( IndexType color = 0; color < this->getNumberOfColors(); color++ )
   {
      this->colorPointers.setElement( color, position );
      for (IndexType i = 0; i < this->getRows(); i++)
         if ( colorsVector.getElement( i ) == color)
         {
            IndexType row1 = this->permutationArray.getElement( i );
            IndexType row2 = this->permutationArray.getElement( position );
            IndexType tmp = this->permutationArray.getElement( row1 );
            this->permutationArray.setElement( row1, this->permutationArray.getElement( row2 ) );
            this->permutationArray.setElement( row2, tmp );

            tmp = colorsVector.getElement( position );
            colorsVector.setElement( position, colorsVector.getElement( i ) );
            colorsVector.setElement( i, tmp );
            position++;
         }
   }

   this->colorPointers.setElement( this->getNumberOfColors(), this->getRows() );

   // destroy colors vector
   colorsVector.reset();

   this->inversePermutationArray.setSize( this->getRows() );
   for( IndexType row = 0; row < this->getRows(); row++ )
      this->inversePermutationArray.setElement( this->permutationArray.getElement( row ), row );
}

template< typename Real,
          typename Device,
          typename Index >
void EllpackSymmetricGraph< Real, Device, Index >::verifyPermutationArray()
{
    for( IndexType i = 0; i < this->getRows(); i++ )
       if( this->permutationArray.getElement( i ) >= this->getRows() )
       {
           std::cerr << "There is wrong data in permutationArray position " << i << std::endl;
           break;
       }
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
__device__ __host__
#endif
bool EllpackSymmetricGraph< Real, Device, Index >::rearrangeMatrix( bool verbose )
{
   // first we need to know permutation
   this->computePermutationArray();
   if( verbose )
      this->verifyPermutationArray();

   // then we need to create new matrix
   Containers::Vector< Real, Device, Index > valuesVector;
   Containers::Vector< Index, Device, Index > columnsVector;
   valuesVector.setSize( this->values.getSize() );
   columnsVector.setSize( this->columnIndexes.getSize() );
   valuesVector.setValue( 0.0 );
   columnsVector.setValue( this->getPaddingIndex() );

   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      typedef EllpackSymmetricGraphDeviceDependentCode< DeviceType > DDCType;
      IndexType elementPtrOrig = DDCType::getRowBegin( *this, row );
      IndexType elementPtrNew = DDCType::getRowBegin( *this, this->permutationArray.getElement( row ) );
      IndexType rowEnd = DDCType::getRowEnd( *this, row );
      IndexType step = DDCType::getElementStep( *this );

      for( IndexType i = 0; i < this->rowLengths; i++ )
      {
         if( this->columnIndexes.getElement( elementPtrOrig ) <= row )
         {
            valuesVector.setElement(elementPtrNew, this->values.getElement(elementPtrOrig));
            columnsVector.setElement(elementPtrNew, this->columnIndexes.getElement(elementPtrOrig));
            elementPtrNew += step;
         }
         elementPtrOrig += step;
      }
   }

   // reset original matrix
   this->values.reset();
   this->columnIndexes.reset();

   // deep copy new matrix
   this->values.setSize( valuesVector.getSize() );
   this->columnIndexes.setSize( columnsVector.getSize() );
   this->values = valuesVector;
   this->columnIndexes = columnsVector;

   // clear memory
   valuesVector.reset();
   columnsVector.reset();

   this->rearranged = true;
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
  __device__ __host__
#endif
Containers::Vector< Index, Device, Index > EllpackSymmetricGraph< Real, Device, Index >::getPermutationArray()
{
    return this->permutationArray;
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
  __device__ __host__
#endif
Containers::Vector< Index, Device, Index > EllpackSymmetricGraph< Real, Device, Index >::getInversePermutation()
{
    return this->inversePermutationArray;
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
  __device__ __host__
#endif
Containers::Vector< Index, Device, Index > EllpackSymmetricGraph< Real, Device, Index >::getColorPointers()
{
    return this->colorPointers;
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
  __device__ __host__
#endif
void EllpackSymmetricGraph< Real, Device, Index >::copyFromHostToCuda( EllpackSymmetricGraph< Real, Devices::Host, Index >& matrix )
{
    //  TODO: fix
    //Sparse< Real, Device, Index >::copyFromHostToCuda( matrix );

    this->rearranged = true;
    this->rowLengths = matrix.getRowLengthsInt();
    this->alignedRows = matrix.getAlignedRows();
    Containers::Vector< Index, Devices::Host, Index > colorPointers = matrix.getColorPointers();
    this->colorPointers.setSize( colorPointers.getSize() );
    for( IndexType i = 0; i < colorPointers.getSize(); i++ )
        this->colorPointers.setElement( i, colorPointers[ i ] );

    Containers::Vector< Index,Devices::Host, Index > permutationArray = matrix.getPermutationArray();
    this->permutationArray.setSize( permutationArray.getSize() );
    for( IndexType i = 0; i < permutationArray.getSize(); i++ )
        this->permutationArray.setElement( i, permutationArray[ i ] );

    Containers::Vector< Index, Devices::Host, Index > inversePermutation = matrix.getInversePermutation();
    this->inversePermutationArray.setSize( inversePermutation.getSize() );
    for( IndexType i = 0; i < inversePermutation.getSize(); i++ )
        this->inversePermutationArray.setElement( i, inversePermutation[ i ] );

    for( IndexType i = 0; i < this->getRows(); i++ )
        for( IndexType j = 0; j <= i; j++ )
            if( matrix.getElement( i, j ) != 0.0 )
                this->setElementFast( i, j, matrix.getElement( i, j ) );

    colorPointers.reset();
    permutationArray.reset();
}

template< typename Real,
          typename Device,
          typename Index >
bool EllpackSymmetricGraph< Real, Device, Index >::setConstantRowLengths( const IndexType& rowLengths )
{
   TNL_ASSERT( rowLengths > 0, std::cerr << " rowLengths = " << rowLengths );
   this->rowLengths = rowLengths;
   if( this->rows > 0 )
      return allocateElements();
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
Index EllpackSymmetricGraph< Real, Device, Index >::getRowLength( const IndexType row ) const
{
   return this->rowLengths;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool EllpackSymmetricGraph< Real, Device, Index >::setLike( const EllpackSymmetricGraph< Real2, Device2, Index2 >& matrix )
{
   if( ! Sparse< Real, Device, Index >::setLike( matrix ) ||
       ! this->permutationArray.setLike( matrix.permutationArray ) ||
       ! this->colorPointers.setLike( matrix.colorPointers ) )
      return false;
   this->rowLengths = matrix.rowLengths;
   this->numberOfColors = matrix.getNumberOfColors();
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
void EllpackSymmetricGraph< Real, Device, Index > :: reset()
{
   Sparse< Real, Device, Index >::reset();
   this->permutationArray.reset();
   this->colorPointers.reset();
   this->rowLengths = 0;
}

/*template< typename Real,
          typename Device,
          typename Index >
   template< typename Matrix >
bool EllpackSymmetricGraph< Real, Device, Index >::copyFrom( const Matrix& matrix,
                                                        const CompressedRowLengthsVector& rowLengths )
{
   return tnlMatrix< RealType, DeviceType, IndexType >::copyFrom( matrix, rowLengths );
}*/

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool EllpackSymmetricGraph< Real, Device, Index > :: setElementFast( const IndexType row,
                                                                     const IndexType column,
                                                                     const Real& value )
{
   return this->addElementFast( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index >
bool EllpackSymmetricGraph< Real, Device, Index > :: setElement( const IndexType row,
                                                                 const IndexType column,
                                                                 const Real& value )
{
   return this->addElement( row, column, value, 0.0 );
}


template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool EllpackSymmetricGraph< Real, Device, Index > :: addElementFast( const IndexType row,
                                                                     const IndexType column,
                                                                     const RealType& value,
                                                                     const RealType& thisElementMultiplicator )
{
   typedef EllpackSymmetricGraphDeviceDependentCode< DeviceType > DDCType;
   IndexType i = DDCType::getRowBegin( *this, this->permutationArray.getElement( row ) );
   const IndexType rowEnd = DDCType::getRowEnd( *this, this->permutationArray.getElement( row ) );
   const IndexType step = DDCType::getElementStep( *this );

   while( i < rowEnd &&
         this->columnIndexes.getElement( i ) < column &&
         this->columnIndexes.getElement( i ) != this->getPaddingIndex() ) i += step;
   if( i == rowEnd )
      return false;
   if( this->columnIndexes.getElement( i ) == column )
   {
      this->values.setElement( i, thisElementMultiplicator * this->values.getElement( i ) + value);
      return true;
   }
   else
      if( this->columnIndexes.getElement( i ) == this->getPaddingIndex() ) // artificial zero
      {
         this->columnIndexes.setElement( i, column);
         this->values.setElement( i, value);
      }
      else
      {
         Index j = rowEnd - step;
         while( j > i )
         {
            this->columnIndexes.setElement( j, this->columnIndexes.getElement( j - step ) );
            this->values.setElement( j, this->values.getElement( j - step ) );
            j -= step;
         }
         this->columnIndexes.setElement( i, column );
         this->values.setElement( i, value );
      }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool EllpackSymmetricGraph< Real, Device, Index > :: addElement( const IndexType row,
                                                                 const IndexType column,
                                                                 const RealType& value,
                                                                 const RealType& thisElementMultiplicator )
{
   typedef EllpackSymmetricGraphDeviceDependentCode< DeviceType > DDCType;
   IndexType i = DDCType::getRowBegin( *this, this->permutationArray[ row ] );
   const IndexType rowEnd = DDCType::getRowEnd( *this, this->permutationArray[ row ] );
   const IndexType step = DDCType::getElementStep( *this );

   while( i < rowEnd &&
          this->columnIndexes.getElement( i ) < column &&
          this->columnIndexes.getElement( i ) != this->getPaddingIndex() ) i += step;
   if( i == rowEnd )
      return false;
   if( this->columnIndexes.getElement( i ) == column )
   {
      this->values.setElement( i, thisElementMultiplicator * this->values.getElement( i ) + value );
      return true;
   }
   else
      if( this->columnIndexes.getElement( i ) == this->getPaddingIndex() )
      {
         this->columnIndexes.setElement( i, column );
         this->values.setElement( i, value );
      }
      else
      {
         IndexType j = rowEnd - step;
         while( j > i )
         {
            this->columnIndexes.setElement( j, this->columnIndexes.getElement( j - step ) );
            this->values.setElement( j, this->values.getElement( j - step ) );
            j -= step;
         }
         this->columnIndexes.setElement( i, column );
         this->values.setElement( i, value );
      }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool EllpackSymmetricGraph< Real, Device, Index > :: setRowFast( const IndexType row,
                                                                 const IndexType* columnIndexes,
                                                                 const RealType* values,
                                                                 const IndexType elements )
{
   typedef EllpackSymmetricGraphDeviceDependentCode< DeviceType > DDCType;
   IndexType elementPointer = DDCType::getRowBegin( *this, this->permutationArray[ row ] );
   const IndexType rowEnd = DDCType::getRowEnd( *this, this->permutationArray[ row ] );
   const IndexType step = DDCType::getElementStep( *this );

   if( elements > this->rowLengths )
      return false;
   for( Index i = 0; i < elements; i++ )
   {
      const IndexType column = columnIndexes[ i ];
      if( column < 0 || column >= this->getColumns() )
         return false;
      this->columnIndexes[ elementPointer ] = column;
      this->values[ elementPointer ] = values[ i ];
      elementPointer += step;
   }
   for( Index i = elements; i < this->rowLengths; i++ )
   {
      this->columnIndexes[ elementPointer ] = this->getPaddingIndex();
      elementPointer += step;
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool EllpackSymmetricGraph< Real, Device, Index > :: setRow( const IndexType row,
                                                             const IndexType* columnIndexes,
                                                             const RealType* values,
                                                             const IndexType elements )
{
   typedef EllpackSymmetricGraphDeviceDependentCode< DeviceType > DDCType;
   IndexType elementPointer = DDCType::getRowBegin( *this, this->permutationArray.getElement( row ) );
   const IndexType rowEnd = DDCType::getRowEnd( *this, this->permutationArray.getElement( row ) );
   const IndexType step = DDCType::getElementStep( *this );

   if( elements > this->rowLengths )
      return false;

   for( IndexType i = 0; i < elements; i++ )
   {
      const IndexType column = columnIndexes[ i ];
      if( column < 0 || column >= this->getColumns() )
         return false;
      this->columnIndexes.setElement( elementPointer, column );
      this->values.setElement( elementPointer, values[ i ] );
      elementPointer += step;
   }
   for( IndexType i = elements; i < this->rowLengths; i++ )
   {
      this->columnIndexes.setElement( elementPointer, this->getPaddingIndex() );
      elementPointer += step;
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
bool EllpackSymmetricGraph< Real, Device, Index > :: addRowFast( const IndexType row,
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
bool EllpackSymmetricGraph< Real, Device, Index > :: addRow( const IndexType row,
                                                             const IndexType* columns,
                                                             const RealType* values,
                                                             const IndexType numberOfElements,
                                                             const RealType& thisElementMultiplicator )
{
   return this->addRowFast( row, columns, values, numberOfElements );
}


template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Real EllpackSymmetricGraph< Real, Device, Index >::getElementFast( const IndexType row,
                                                                   const IndexType column ) const
{
   if( row < column )
       return this->getElementFast( column, row );

   typedef EllpackSymmetricGraphDeviceDependentCode< DeviceType > DDCType;
   IndexType elementPtr = DDCType::getRowBegin( *this, this->permutationArray.getElement( row ) );
   const IndexType rowEnd = DDCType::getRowEnd( *this, this->permutationArray.getElement( row ) );
   const IndexType step = DDCType::getElementStep( *this );

   while( elementPtr < rowEnd &&
          this->columnIndexes.getElement( elementPtr ) < column &&
          this->columnIndexes.getElement( elementPtr ) != this->getPaddingIndex() ) elementPtr += step;
   if( elementPtr < rowEnd && this->columnIndexes.getElement( elementPtr ) == column )
      return this->values.getElement( elementPtr );
   return 0.0;
}

template< typename Real,
          typename Device,
          typename Index >
Real EllpackSymmetricGraph< Real, Device, Index >::getElement( const IndexType row,
                                                               const IndexType column ) const
{
   if( row < column )
      return this->getElement( column, row );

   typedef EllpackSymmetricGraphDeviceDependentCode< DeviceType > DDCType;
   IndexType elementPtr = DDCType::getRowBegin( *this, this->permutationArray.getElement( row ) );
   const IndexType rowEnd = DDCType::getRowEnd( *this, this->permutationArray.getElement( row ) );
   const IndexType step = DDCType::getElementStep( *this );

   while( elementPtr < rowEnd &&
          this->columnIndexes.getElement( elementPtr ) < column &&
          this->columnIndexes.getElement( elementPtr ) != this->getPaddingIndex() )
   {
      elementPtr += step;
   }
   if( elementPtr < rowEnd && this->columnIndexes.getElement( elementPtr ) == column )
      return this->values.getElement( elementPtr );
   return 0.0;
}


template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
void EllpackSymmetricGraph< Real, Device, Index >::getRowFast( const IndexType row,
                                                               IndexType* columns,
                                                               RealType* values ) const
{
   typedef EllpackSymmetricGraphDeviceDependentCode< DeviceType > DDCType;
   IndexType elementPtr = DDCType::getRowBegin( *this, this->permutationArray[ row ] );
   const IndexType rowEnd = DDCType::getRowEnd( *this, this->permutationArray[ row ] );
   const IndexType step = DDCType::getElementStep( *this );

   for( IndexType i = 0; i < this->rowLengths; i++ )
   {
      columns[ i ] = this->columnIndexes[ elementPtr ];
      values[ i ] = this->values[ elementPtr ];
      elementPtr += step;
   }
}

template< typename Real,
          typename Device,
          typename Index >
void EllpackSymmetricGraph< Real, Device, Index >::getRow( const IndexType row,
                                                           IndexType* columns,
                                                           RealType* values ) const
{
   typedef EllpackSymmetricGraphDeviceDependentCode< DeviceType > DDCType;
   IndexType elementPtr = DDCType::getRowBegin( *this, this->permutationArray[ row ] );
   const IndexType rowEnd = DDCType::getRowEnd( *this, this->permutationArray[ row ] );
   const IndexType step = DDCType::getElementStep( *this );

   for( IndexType i = 0; i < this->rowLengths; i++ )
   {
      columns[ i ] = this->columnIndexes.getElement( elementPtr );
      values[ i ] = this->values.getElement( elementPtr );
      elementPtr += step;
   }
}

template< typename Real,
          typename Device,
          typename Index >
  template< typename Vector >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
typename Vector::RealType EllpackSymmetricGraph< Real, Device, Index >::rowVectorProduct( const IndexType row,
                                                                                          const Vector& vector ) const
{
   IndexType i = DeviceDependentCode::getRowBegin( *this, row );
   const IndexType rowEnd = DeviceDependentCode::getRowEnd( *this, row );
   const IndexType step = DeviceDependentCode::getElementStep( *this );

   Real result = 0.0;
   while( i < rowEnd && this->columnIndexes[ i ] != this->getPaddingIndex() )
   {
      const Index column = this->columnIndexes[ i ];
      result += this->values[ i ] * vector[ column ];
      i += step;
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename InVector,
             typename OutVector >
void EllpackSymmetricGraph< Real, Device, Index >::vectorProduct( const InVector& inVector,
                                                                  OutVector& outVector ) const
{
   DeviceDependentCode::vectorProduct( *this, inVector, outVector );
}

template< typename Real,
          typename Device,
          typename Index >
bool EllpackSymmetricGraph< Real, Device, Index >::save( File& file ) const
{
   if( ! Sparse< Real, Device, Index >::save( file) ) return false;
#ifdef HAVE_NOT_CXX11
   if( ! file.write< IndexType, Devices::Host, IndexType >( &this->rowLengths, 1 ) ) return false;
#else      
   if( ! file.write( &this->rowLengths ) ) return false;
#endif   
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool EllpackSymmetricGraph< Real, Device, Index >::load( File& file )
{
   if( ! Sparse< Real, Device, Index >::load( file) ) return false;
#ifdef HAVE_NOT_CXX11
   if( ! file.read< IndexType, Devices::Host, IndexType >( &this->rowLengths, 1 ) ) return false;
#else   
   if( ! file.read( &this->rowLengths ) ) return false;
#endif   
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool EllpackSymmetricGraph< Real, Device, Index >::save( const String& fileName ) const
{
   return Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
bool EllpackSymmetricGraph< Real, Device, Index >::load( const String& fileName )
{
   return Object::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
bool EllpackSymmetricGraph< Real, Device, Index >::help( bool verbose )
{
    if( !this->rearranged )
        return this->rearrangeMatrix( verbose );
}

template< typename Real,
          typename Device,
          typename Index >
void EllpackSymmetricGraph< Real, Device, Index >::print( std::ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";
      IndexType i( row * this->rowLengths );
      const IndexType rowEnd( i + this->rowLengths );
      while( i < rowEnd &&
             this->columnIndexes.getElement( i ) < this->columns &&
             this->columnIndexes.getElement( i ) != this->getPaddingIndex() )
      {
         const Index column = this->columnIndexes.getElement( i );
         str << " Col:" << column << "->" << this->values.getElement( i ) << "\t";
         i++;
      }
      str << std::endl;
   }
}

template< typename Real,
          typename Device,
          typename Index >
bool EllpackSymmetricGraph< Real, Device, Index >::allocateElements()
{
   Sparse< Real, Device, Index >::allocateMatrixElements( this->alignedRows * this->rowLengths );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename InVector,
          typename OutVector >
void EllpackSymmetricGraph< Real, Device, Index >::vectorProductHost( const InVector& inVector,
                                                                      OutVector& outVector ) const
{
   for( IndexType color = 0; color < this->getNumberOfColors(); color++ )
   {
      // IndexType colorBegin = this->colorPointers[ color ];
      IndexType offset = this->colorPointers[ color ];
      IndexType colorEnd = this->colorPointers[ color + 1 ];
      for( IndexType j = 0; j < this->getRowsOfColor( color ); j++ )
      {
         IndexType row = offset + j;
         if( row >= colorEnd )
            break;
         IndexType i = DeviceDependentCode::getRowBegin( *this, row );
         const IndexType rowEnd = DeviceDependentCode::getRowEnd( *this, row );
         const IndexType step = DeviceDependentCode::getElementStep( *this );
         const IndexType rowMapping = this->inversePermutationArray[ row ];

         while( i < rowEnd && this->columnIndexes[ i ] != this->getPaddingIndex() )
         {
            const IndexType column = this->columnIndexes[ i ];
            outVector[ rowMapping ] += this->values[ i ] * inVector[ column ];
            if( rowMapping != column )
               outVector[ column ] += this->values[ i ] * inVector[ rowMapping ];
            i += step;
         }
      }
   }
}

template<>
class EllpackSymmetricGraphDeviceDependentCode< Devices::Host >
{
   public:

      typedef Devices::Host Device;

      template< typename Real,
                typename Index >
      static Index getRowBegin( const EllpackSymmetricGraph< Real, Device, Index >& matrix,
                                const Index row )
      {
         return row * matrix.rowLengths;
      }

      template< typename Real,
                typename Index >
      static Index getRowEnd( const EllpackSymmetricGraph< Real, Device, Index >& matrix,
                                const Index row )
      {
         return ( row + 1 ) * matrix.rowLengths;
      }

      template< typename Real,
                typename Index >
      static Index getElementStep( const EllpackSymmetricGraph< Real, Device, Index >& matrix )
      {
         return 1;
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const EllpackSymmetricGraph< Real, Device, Index >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         matrix.vectorProductHost( inVector, outVector );
      }
};

#ifdef HAVE_CUDA
template< typename Real,
          typename Device,
          typename Index >
template< typename InVector,
          typename OutVector >
__device__
void EllpackSymmetricGraph< Real, Device, Index >::spmvCuda( const InVector& inVector,
                                                             OutVector& outVector,
                                                             const int globalIdx,
                                                             const int color ) const
{
   IndexType offset = this->colorPointers[ color ];
   const IndexType colorEnd = this->colorPointers[ color + 1 ];
   IndexType row = offset + globalIdx;
   if( row >= colorEnd )
      return;

   IndexType i = DeviceDependentCode::getRowBegin( *this, row );
   const IndexType rowEnd = DeviceDependentCode::getRowEnd( *this, row );
   const IndexType step = DeviceDependentCode::getElementStep( *this );
   const IndexType rowMapping = this->inversePermutationArray[ row ];

   while( i < rowEnd && this->columnIndexes[ i ] != this->getPaddingIndex() )
   {
      const IndexType column = this->columnIndexes[ i ];
      outVector[ rowMapping ] += this->values[ i ] * inVector[ column ];
      if( rowMapping != column )
         outVector[ column ] += this->values[ i ] * inVector[ rowMapping ];
      i += step;
   }
}
#endif

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename InVector,
          typename OutVector >
__global__
void EllpackSymmetricGraphVectorProductCuda( const EllpackSymmetricGraph< Real, Devices::Cuda, Index >* matrix,
                                             const InVector* inVector,
                                             OutVector* outVector,
                                             const int gridIdx,
                                             const int color )
{
   int globalIdx = ( gridIdx * Devices::Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   matrix->spmvCuda( *inVector, *outVector, globalIdx, color );
}
#endif

template<>
class EllpackSymmetricGraphDeviceDependentCode< Devices::Cuda >
{
   public:

      typedef Devices::Cuda Device;

      template< typename Real,
                typename Index >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
      static Index getRowBegin( const EllpackSymmetricGraph< Real, Device, Index >& matrix,
                                const Index row )
      {
         return row;
      }

      template< typename Real,
                typename Index >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
      static Index getRowEnd( const EllpackSymmetricGraph< Real, Device, Index >& matrix,
                                const Index row )
      {
         return row + getElementStep( matrix ) * matrix.rowLengths;
      }

      template< typename Real,
                typename Index >
#ifdef HAVE_CUDA
      __device__ __host__
#endif
      static Index getElementStep( const EllpackSymmetricGraph< Real, Device, Index >& matrix )
      {
         return matrix.alignedRows;
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const EllpackSymmetricGraph< Real, Device, Index >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
#ifdef HAVE_CUDA
          typedef EllpackSymmetricGraph< Real, Devices::Cuda, Index > Matrix;
          typedef typename Matrix::IndexType IndexType;
          Matrix* kernel_this = Devices::Cuda::passToDevice( matrix );
          InVector* kernel_inVector = Devices::Cuda::passToDevice( inVector );
          OutVector* kernel_outVector = Devices::Cuda::passToDevice( outVector );
          dim3 cudaBlockSize( 256 ), cudaGridSize( Devices::Cuda::getMaxGridSize() );
          for( IndexType color = 0; color < matrix.getNumberOfColors(); color++ )
          {
              IndexType rows = matrix.getRowsOfColor( color );
              const IndexType cudaBlocks = roundUpDivision( rows, cudaBlockSize.x );
              const IndexType cudaGrids = roundUpDivision( cudaBlocks, Devices::Cuda::getMaxGridSize() );
              for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
              {
                  if( gridIdx == cudaGrids - 1 )
                      cudaGridSize.x = cudaBlocks % Devices::Cuda::getMaxGridSize();
                  EllpackSymmetricGraphVectorProductCuda< Real, Index, InVector, OutVector >
                                                      <<< cudaGridSize, cudaBlockSize >>>
                                                        ( kernel_this,
                                                          kernel_inVector,
                                                          kernel_outVector,
                                                          gridIdx,
                                                          color );
              }
          }

          Devices::Cuda::freeFromDevice( kernel_this );
          Devices::Cuda::freeFromDevice( kernel_inVector );
          Devices::Cuda::freeFromDevice( kernel_outVector );
          TNL_CHECK_CUDA_DEVICE;
#endif
      }
};

} // namespace Matrices
} // namespace TNL
