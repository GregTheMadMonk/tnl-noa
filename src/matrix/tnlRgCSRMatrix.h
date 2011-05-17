/***************************************************************************
                          tnlRgCSRMatrix.h  -  description
                             -------------------
    begin                : Jul 10, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
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


#ifndef TNLRGCSRMATRIX_H
#define TNLRGCSRMATRIX_H

#include <iostream>
#include <iomanip>
#include <core/tnlLongVectorHost.h>
#include <core/tnlAssert.h>
#include <core/mfuncs.h>
#include <matrix/tnlCSRMatrix.h>
#include <debug/tnlDebug.h>

using namespace std;

enum tnlAdaptiveGroupSizeStrategy { tnlAdaptiveGroupSizeStrategyByAverageRowSize,
                                    tnlAdaptiveGroupSizeStrategyByFirstGroup };

//! Matrix storing the non-zero elements in the Row-grouped CSR (Compressed Sparse Row) format
/*!
 */
template< typename Real, tnlDevice Device = tnlHost, typename Index = int  >
class tnlRgCSRMatrix : public tnlMatrix< Real, Device, Index >
{
   public:
   //! Basic constructor
   tnlRgCSRMatrix( const tnlString& name );

   const tnlString& getMatrixClass() const;

   tnlString getType() const;

   Index getGroupSize() const;

   Index getCUDABlockSize() const;

   //! This can only be a multiple of the groupSize
   void setCUDABlockSize( Index blockSize );

   //! Sets the number of row and columns.
   bool setSize( Index new_size );

   //! Allocate memory for the nonzero elements.
   bool setNonzeroElements( Index elements );

   Index getNonzeroElements() const;

   Index getArtificialZeroElements() const;

   bool setElement( Index row,
                    Index colum,
                    const Real& value )
   { abort(); };

   bool addToElement( Index row,
                      Index column,
                      const Real& value )
   { abort(); };

   bool copyFrom( const tnlCSRMatrix< Real, tnlHost, Index >& csr_matrix,
                  const Index groupSize,
                  const bool useAdaptiveGroupSize = false,
                  const tnlAdaptiveGroupSizeStrategy adaptiveGroupSizeStrategy = tnlAdaptiveGroupSizeStrategyByAverageRowSize );

   template< tnlDevice Device2 >
   bool copyFrom( const tnlRgCSRMatrix< Real, Device2, Index >& rgCSRMatrix );

   Real getElement( Index row,
                    Index column ) const;

   Real rowProduct( Index row,
                    const tnlLongVector< Real, Device, Index >& vector ) const;

   void vectorProduct( const tnlLongVector< Real, Device, Index >& x,
                       tnlLongVector< Real, Device, Index >& b ) const;

   Real getRowL1Norm( Index row ) const
   { abort(); };

   void multiplyRow( Index row, const Real& value )
   { abort(); };

   //! Prints out the matrix structure
   void printOut( ostream& str,
		          const Index lines = 0 ) const;

   protected:

   /****
    * Returns group size of the group with given ID.
    * Group size may vary when useAdaptiveGroupSize == true. If not the last group in the
    * matrix may also be smaller then the parameter groupSize.
    */
   Index getCurrentGroupSize( const Index groupId ) const;

   //! Insert one block to the matrix.
   /*!**
    *  If there is some data already in this @param row it will be rewritten.
    *  @param elements says number of non-zero elements which will be inserted.
    *  @param data is pointer to the elements values.
    *  @param first_column is the column of the first non-zero element.
    *  @param offsets is a pointer to field with offsets of the elements with
    *  respect to the first one. All of them must sorted increasingly.
    *  The elements which do not fit to the matrix are omitted.
    */
   bool insertBlock( );

   bool useAdaptiveGroupSize;

   tnlLongVector< Real, Device, Index > nonzeroElements;

   tnlLongVector< Index, Device, Index > columns;

   tnlLongVector< Index, Device, Index > groupOffsets;

   tnlLongVector< Index, Device, Index > nonzeroElementsInRow;

   Index groupSize;

   /****
    * This vector is only used if useAdaptiveGroupSize is true.
    */
   tnlLongVector< Index, Device, Index > adaptiveGroupSizes;

   /****
    * This variable is only used if useAdaptiveGroupSize is true.
    */
   Index numberOfGroups;

   Index cudaBlockSize;

   Index artificial_zeros;

   //! The last non-zero element is at the position last_non_zero_element - 1
   Index last_nonzero_element;

   friend class tnlRgCSRMatrix< Real, tnlHost, Index >;
   friend class tnlRgCSRMatrix< Real, tnlCuda, Index >;
};

#ifdef HAVE_CUDA
/****
 *  The CUDA documentation says: "A texture reference is declared at file scope as a variable of type texture".
 *  It cannot be passed as a reference otherwise we get ptx error:
 *   State space mismatch between instruction and address in instruction 'tex'.
 *  Therefore we keep texture as global variable handled by tnlCudaTextureBinder
 */
texture< float, 1 > tnlRgCSRMatrixCUDA_floatTexRef;
texture< int2, 1 > tnlRgCSRMatrixCUDA_doubleTexRef;

bool bindRgCSRMatrixCUDATexture( const float* data,
                                        size_t size );

bool bindRgCSRMatrixCUDATexture( const double* data,
                                        size_t size );

/****
 * The pointer here is dummy. It is only for knowing what type of texture we want to unbind - float or double.
 */
bool __inline__ unbindRgCSRMatrixCUDATexture( const float* dummy_pointer );

/****
 * The pointer here is dummy. It is only for knowing what type of texture we want to unbind - float or double.
 */
bool __inline__ unbindRgCSRMatrixCUDATexture( const double* dummy_pointer );

template< bool UseCache, typename Index >
static __inline__ __device__ float fetchVecX( const Index i,
                                              const float* x );

template< bool UseCache, typename Index >
static __inline__ __device__ double fetchVecX( const Index i,
                                               const double* x );

template< class Real, typename Index, bool useCache >
__global__ void sparseOldCSRMatrixVectorProductKernel( Index size,
                                                       Index block_size,
                                                       const Real* nonzeroElements,
                                                       const Index* columns,
                                                       const Index* groupOffsets,
                                                       const Index* nonzeroElementsInRow,
                                                       const Real* vec_x,
                                                       Real* vec_b );

template< class Real, typename Index, bool useCache >
__global__ void sparseCSRMatrixVectorProductKernel( Index size,
                                                    Index groupSize,
                                                    const Real* nonzeroElements,
                                                    const Index* columns,
                                                    const Index* groupOffsets,
                                                    const Index* nonzerosInRow,
                                                    const Real* vec_x,
                                                    Real* vec_b );
#endif


template< typename Real, tnlDevice Device, typename Index >
tnlRgCSRMatrix< Real, Device, Index > :: tnlRgCSRMatrix( const tnlString& name )
: tnlMatrix< Real, Device, Index >( name ),
  useAdaptiveGroupSize( false ),
  nonzeroElements( name + " : nonzeroElements" ),
  columns( name + " : columns" ),
  groupOffsets( name + " : block-offsets" ),
  nonzeroElementsInRow( name + " : nonzerosInRow" ),
  groupSize( 0 ),
  adaptiveGroupSizes( name + "adaptiveGroupSizes" ),
  numberOfGroups( 0 ),
  cudaBlockSize( 0 ),
  artificial_zeros( 0 ),
  last_nonzero_element( 0 )
{
};

template< typename Real, tnlDevice Device, typename Index >
const tnlString& tnlRgCSRMatrix< Real, Device, Index > :: getMatrixClass() const
{
   return tnlMatrixClass :: main;
};

template< typename Real, tnlDevice Device, typename Index >
tnlString tnlRgCSRMatrix< Real, Device, Index > :: getType() const
{
   return tnlString( "tnlRgCSRMatrix< ") +
          tnlString( GetParameterType( Real( 0.0 ) ) ) +
          tnlString( ", " ) +
          getDeviceType( Device ) +
          tnlString( ", " ) +
          GetParameterType( Index( 0 ) ) +
          tnlString( " >" );
   // TODO: add value of useAdaptiveGroupSize
};

template< typename Real, tnlDevice Device, typename Index >
Index tnlRgCSRMatrix< Real, Device, Index > :: getGroupSize() const
{
   return groupSize;
}

template< typename Real, tnlDevice Device, typename Index >
Index tnlRgCSRMatrix< Real, Device, Index > :: getCUDABlockSize() const
{
   return cudaBlockSize;
}

template< typename Real, tnlDevice Device, typename Index >
void tnlRgCSRMatrix< Real, Device, Index > :: setCUDABlockSize( Index blockSize )
{
   tnlAssert( blockSize % this -> getGroupSize() == 0, )
   cudaBlockSize = blockSize;
}

template< typename Real, tnlDevice Device, typename Index >
bool tnlRgCSRMatrix< Real, Device, Index > :: setSize( Index new_size )
{
   this -> size = new_size;
   if( ! groupOffsets. setSize( this -> getSize() / groupSize + ( this -> getSize() % groupSize != 0 ) + 1 ) ||
	    ! nonzeroElementsInRow. setSize( this -> getSize() ) ||
	    ! adaptiveGroupSizes. setSize( this -> getSize() ) )
      return false;
   groupOffsets. setValue( 0 );
   nonzeroElementsInRow. setValue( 0 );
   adaptiveGroupSizes. setValue( 0 );
   last_nonzero_element = 0;
   return true;
};

template< typename Real, tnlDevice Device, typename Index >
bool tnlRgCSRMatrix< Real, Device, Index > :: setNonzeroElements( Index elements )
{
   if( ! nonzeroElements. setSize( elements ) ||
	    ! columns. setSize( elements ) )
      return false;
   nonzeroElements. setValue( 0.0 );
   columns. setValue( -1 );
   return true;
};

template< typename Real, tnlDevice Device, typename Index >
Index tnlRgCSRMatrix< Real, Device, Index > :: getNonzeroElements() const
{
   tnlAssert( nonzeroElements. getSize() > artificial_zeros, );
	return nonzeroElements. getSize() - artificial_zeros;
}

template< typename Real, tnlDevice Device, typename Index >
Index tnlRgCSRMatrix< Real, Device, Index > :: getArtificialZeroElements() const
{
	return artificial_zeros;
}

template< typename Real, tnlDevice Device, typename Index >
bool tnlRgCSRMatrix< Real, Device, Index > :: copyFrom( const tnlCSRMatrix< Real, tnlHost, Index >& csr_matrix,
                                                        const Index groupSize,
                                                        const bool useAdaptiveGroupSize,
                                                        const tnlAdaptiveGroupSizeStrategy adaptiveGroupSizeStrategy )
{
	dbgFunctionName( "tnlRgCSRMatrix< Real, tnlHost >", "copyFrom" );
	this -> groupSize = groupSize;
	tnlAssert( this -> groupSize > 0, );
	this -> useAdaptiveGroupSize = useAdaptiveGroupSize;

	if( ! this -> setSize( csr_matrix. getSize() ) )
		return false;
	dbgExpr( csr_matrix. getSize() );

	/****
	 * In case of adaptive group sizes compute maximum number of the non-zero elements in group.
	 */
	Index maxNonzeroElementsInGroup( 0 );
	if( useAdaptiveGroupSize )
	{
	   if( adaptiveGroupSizeStrategy == tnlAdaptiveGroupSizeStrategyByAverageRowSize )
	   {
	      const Index averageRowSize = ceil( ( float ) csr_matrix. getNonzeroElements() / ( float ) csr_matrix. getSize() );
	      maxNonzeroElementsInGroup = averageRowSize * groupSize;
	   }
	   if( adaptiveGroupSizeStrategy == tnlAdaptiveGroupSizeStrategyByFirstGroup )
	      for( Index row = 0; row < groupSize; row ++ )
	         maxNonzeroElementsInGroup += csr_matrix. getNonzeroElementsInRow( row );
	}

	/****
	 *  Now compute the number of non-zero elements in each row
	 *  and compute number of elements which are necessary allocate.
	 */
	Index total_elements( 0 );
	Index max_row_in_block( 0 );
	Index currentGroupSize( 0 );
	Index nonzeroElementsInGroup( 0 );
	Index processedLines( 0 );
	numberOfGroups = 0;
	groupOffsets[ 0 ] = 0;
	for( Index i = 0; i < this -> getSize(); i ++ )
	{
		if( i > 0 && i % groupSize == 0 )
		{
		   currentGroupSize += groupSize;
		   if( ! useAdaptiveGroupSize || nonzeroElementsInGroup > maxNonzeroElementsInGroup )
		   {
		      if( useAdaptiveGroupSize )
		         adaptiveGroupSizes[ numberOfGroups ] = currentGroupSize;

		      dbgCout( numberOfGroups << "-th group size is " << currentGroupSize );
		      dbgCout( "Elements in this group " << max_row_in_block * currentGroupSize );
		      total_elements += max_row_in_block * currentGroupSize;
		      groupOffsets[ numberOfGroups + 1 ] = total_elements;
		      dbgCout( "Total elements so far " << total_elements );

		      numberOfGroups ++;
		      processedLines += currentGroupSize;
		      max_row_in_block = 0;
		      currentGroupSize = 0;
		      nonzeroElementsInGroup = 0;
		   }
		}
		nonzeroElementsInRow[ i ] = csr_matrix. row_offsets[ i + 1 ] - csr_matrix. row_offsets[ i ];
		nonzeroElementsInGroup += nonzeroElementsInRow[ i ];
		//dbgExpr( nonzeroElementsInRow[ i ] );
		max_row_in_block = Max( max_row_in_block, nonzeroElementsInRow[ i ] );

	}
	dbgExpr( processedLines );
	currentGroupSize = this -> getSize() - processedLines;
	total_elements += max_row_in_block * ( currentGroupSize );
	groupOffsets[ numberOfGroups + 1 ] = total_elements;
   if( useAdaptiveGroupSize )
       adaptiveGroupSizes[ numberOfGroups ] = currentGroupSize;
	dbgCout( numberOfGroups << "-th group size is " << currentGroupSize );
	numberOfGroups ++;


	/****
	 * Allocate the non-zero elements (they contains some artificial zeros.)
	 */
	dbgCout( "Allocating " << total_elements << " elements.");
	if( ! setNonzeroElements( total_elements ) )
		return false;
	artificial_zeros = total_elements - csr_matrix. getNonzeroElements();

	dbgCout( "Inserting data " );
	if( Device == tnlHost )
	{
	   Index elementRow( 0 );
      /***
       * Insert the data into the groups.
       * We go through the groups.
       */
      for( Index groupId = 0; groupId < numberOfGroups; groupId ++ )
      {
         Index currentGroupSize = getCurrentGroupSize( groupId );
         dbgExpr( groupOffsets[ groupId ] );
         dbgExpr( currentGroupSize );

         /****
          * We insert 'currentGroupSize' rows in this matrix with the stride
          * given by the block size.
          */
         for( Index k = 0; k < currentGroupSize; k ++ )
         {
            /****
             * We start with the offset k within the block and
             * we insert the data with a stride equal to the block size.
             * j - is the element position in the nonzeroElements in this matrix
             */
            Index j = groupOffsets[ groupId ] + k;                   // position of the first element of the row
            //dbgExpr( element_row );
            if( elementRow < this -> getSize() )
            {

               /****
                * Get the element position
                */
               Index elementPos = csr_matrix. row_offsets[ elementRow ];
               while( elementPos < csr_matrix. row_offsets[ elementRow + 1 ] )
               {
                  dbgCout( "Inserting on position " << j
                         << " data " << csr_matrix. nonzero_elements[ elementPos ]
                         << " at column " << csr_matrix. columns[ elementPos ] );
                  nonzeroElements[ j ] = csr_matrix. nonzero_elements[ elementPos ];
                  columns[ j ] = csr_matrix. columns[ elementPos ];
                  elementPos ++;
                  j += currentGroupSize;
               }
               elementRow ++;
            }
         }
      }
	}
	if( Device == tnlCuda )
	{
	   tnlAssert( false,
	              cerr << "Conversion from tnlCSRMatrix on the host to the tnlRgCSRMatrix on the CUDA device is not implemented yet."; );
	   //TODO: implement this
	}
	return true;
};

template< typename Real, tnlDevice Device, typename Index >
   template< tnlDevice Device2 >
bool tnlRgCSRMatrix< Real, Device, Index > :: copyFrom( const tnlRgCSRMatrix< Real, Device2, Index >& rgCSRMatrix )
{
   dbgFunctionName( "tnlRgCSRMatrix< Real, Device, Index >", "copyFrom" );
   groupSize = rgCSRMatrix. getGroupSize();
   if( ! this -> setSize( rgCSRMatrix. getSize() ) )
      return false;

   /****
    * Allocate the non-zero elements (they contains some artificial zeros.)
    */
   Index total_elements = rgCSRMatrix. getNonzeroElements() + 
                          rgCSRMatrix. getArtificialZeroElements() ;
   dbgCout( "Allocating " << total_elements << " elements.");
   if( ! setNonzeroElements( total_elements ) )
      return false;
   artificial_zeros = total_elements - rgCSRMatrix. getNonzeroElements();

   nonzeroElements = rgCSRMatrix. nonzeroElements;
   columns = rgCSRMatrix. columns;
   groupOffsets = rgCSRMatrix. groupOffsets;
   nonzeroElementsInRow = rgCSRMatrix. nonzeroElementsInRow;
   last_nonzero_element = rgCSRMatrix. last_nonzero_element;

   return true;
};


template< typename Real, tnlDevice Device, typename Index >
Real tnlRgCSRMatrix< Real, Device, Index > :: getElement( Index row,
                                                          Index column ) const
{
	tnlAssert( 0 <= row && row < this -> getSize(),
			   cerr << "The row is outside the matrix." );
	if( Device == tnlHost )
	{
      Index groupId = row / groupSize;
      Index groupRow = row % groupSize;
      Index groupOffset = groupOffsets[ groupId ];
      /****
       * The last block may be smaller then the global groupSize.
       * We store it in the current_groupSize
       */
      Index currentGroupSize = groupSize;
      if( ( groupId + 1 ) * groupSize > this -> getSize() )
         currentGroupSize = this -> getSize() % groupSize;
      Index pos = groupOffset + groupRow;
      for( Index i = 0; i < nonzeroElementsInRow[ row ]; i ++ )
      {
         if( columns[ pos ] == column )
            return nonzeroElements[ pos ];
         pos += currentGroupSize;
      }
      return Real( 0.0 );
	}
	if( Device == tnlCuda )
	{
	   tnlAssert( false,
	             cerr << "tnlRgCSRMatrix< Real, tnlCuda, Index > ::getElement is not implemented yet." );
	   //TODO: implement this

	}
}

template< typename Real, tnlDevice Device, typename Index >
Real tnlRgCSRMatrix< Real, Device, Index > :: rowProduct( Index row,
                                                          const tnlLongVector< Real, Device, Index >& vec ) const
{
   tnlAssert( 0 <= row && row < this -> getSize(),
              cerr << "The row is outside the matrix." );
   tnlAssert( vec. getSize() == this -> getSize(),
              cerr << "The matrix and vector for multiplication have different sizes. "
                   << "The matrix size is " << this -> getSize() << "."
                   << "The vector size is " << vec. getSize() << endl; );

   if( Device == tnlHost )
   {
      tnlAssert( false, );
      /****
       * TODO: fix this
       * The groupId is not correct if the group size is variable
       */
      Index groupId = row / groupSize;
      Index groupRow = row % groupSize;
      Index groupOffset = groupOffsets[ groupId ];
      Index currentGroupSize = getCurrentGroupSize( groupId );
      Real product( 0.0 );
      Index pos = groupOffset + groupRow;
      for( Index i = 0; i < nonzeroElementsInRow[ row ]; i ++ )
      {
         tnlAssert( pos < nonzeroElements. getSize(), );
         product += nonzeroElements[ pos ] * vec[ columns[ pos ] ];
         pos += currentGroupSize;
      }
      return product;
   }
   if( Device == tnlCuda )
   {
      tnlAssert( false,
               cerr << "tnlRgCSRMatrix< Real, tnlCuda > :: rowProduct is not implemented yet." );
      //TODO: implement this
   }
}

template< typename Real, tnlDevice Device, typename Index >
void tnlRgCSRMatrix< Real, Device, Index > :: vectorProduct( const tnlLongVector< Real, Device, Index >& vec,
                                                             tnlLongVector< Real, Device, Index >& result ) const
{
   dbgFunctionName( "tnlRgCSRMatrix< Real, tnlHost >", "vectorProduct" )
   tnlAssert( vec. getSize() == this -> getSize(),
              cerr << "The matrix and vector for a multiplication have different sizes. "
                   << "The matrix size is " << this -> getSize() << "."
                   << "The vector size is " << vec. getSize() << endl; );
   tnlAssert( result. getSize() == this -> getSize(),
              cerr << "The matrix and result vector of a multiplication have different sizes. "
                   << "The matrix size is " << this -> getSize() << "."
                   << "The vector size is " << vec. getSize() << endl; );

   if( Device == tnlHost )
   {
      const Index blocks_num = groupOffsets. getSize() - 1;
      const Index* row_lengths = nonzeroElementsInRow. getVector();
      const Real* values = nonzeroElements. getVector();
      const Index* cols = columns. getVector();
      const Index* groupOffset = groupOffsets. getVector();
      Index firstRowOfGroup = 0;
      for( Index groupId = 0; groupId < numberOfGroups; groupId ++ )
      {
         dbgExpr( groupId );
         /****
          * The last block may be smaller then the global groupSize.
          * We store it in the current_groupSize
          */
         Index currentGroupSize = getCurrentGroupSize( groupId );

         dbgExpr( currentGroupSize );

         Index block_begining = groupOffset[ groupId ];
         const Index block_length = groupOffset[ groupId + 1 ] - block_begining;
         const Index max_row_length = block_length / currentGroupSize;
         const Index first_row = groupId * groupSize;

         Index csr_col = 0;
         Index row = firstRowOfGroup;
         for( Index block_row = 0; block_row < currentGroupSize; block_row ++ )
         {
            //const Index row = first_row + block_row;
            result[ row ] = 0.0;
            if( csr_col < row_lengths[ row ] )
               result[ row ] += values[ block_begining ] * vec[ cols[ block_begining ] ];
            block_begining ++;
            row ++;
         }
         for( Index csr_col = 1; csr_col < max_row_length; csr_col ++ )
         {
            row = firstRowOfGroup;
            for( Index block_row = 0; block_row < currentGroupSize; block_row ++ )
            {
               //const Index row = first_row + block_row;
               if( csr_col < row_lengths[ row ] )
                  result[ row ] += values[ block_begining ] * vec[ cols[ block_begining ] ];
               block_begining ++;
               row ++;
            }
         }
         firstRowOfGroup += currentGroupSize;
      }
   }
   if( Device == tnlCuda )
   {
#ifdef HAVE_CUDA
   Index blockSize = this -> getCUDABlockSize();
   if( ! blockSize )
      blockSize = this -> getGroupSize();
   const Index size = this -> getSize();

   bool useCache = bindRgCSRMatrixCUDATexture( vec. getVector(),
                                               vec. getSize() );

   cudaThreadSynchronize();
   int gridSize = size / blockSize + ( size % blockSize != 0 ) + 1;
   dim3 gridDim( gridSize ), blockDim( blockSize );
   if( useCache )
      sparseOldCSRMatrixVectorProductKernel< Real, Index, true ><<< gridDim, blockDim >>>( size,
                                                                                        this -> getGroupSize(),
                                                                                        nonzeroElements. getVector(),
                                                                                        columns. getVector(),
                                                                                        groupOffsets. getVector(),
                                                                                        nonzeroElementsInRow. getVector(),
                                                                                        vec. getVector(),
                                                                                        result. getVector() );
   else
      sparseOldCSRMatrixVectorProductKernel< Real, Index, false ><<< gridDim, blockDim >>>( size,
                                                                                        this -> getGroupSize(),
                                                                                        nonzeroElements. getVector(),
                                                                                        columns. getVector(),
                                                                                        groupOffsets. getVector(),
                                                                                        nonzeroElementsInRow. getVector(),
                                                                                        vec. getVector(),
                                                                                        result. getVector() );
    cudaThreadSynchronize();
    unbindRgCSRMatrixCUDATexture( vec. getVector() );
    CHECK_CUDA_ERROR;
#else
   cerr << "CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif
   }

}

template< typename Real, tnlDevice Device, typename Index >
void tnlRgCSRMatrix< Real, Device, Index > :: printOut( ostream& str,
		                                                  const Index lines ) const
{
   str << "Structure of tnlRgCSRMatrix" << endl;
   str << "Matrix name:" << this -> getName() << endl;
   str << "Matrix size:" << this -> getSize() << endl;
   str << "Allocated elements:" << nonzeroElements. getSize() << endl;
   str << "Matrix blocks: " << groupOffsets. getSize() << endl;

   Index print_lines = lines;
   if( ! print_lines )
	   print_lines = this -> getSize();

   for( Index i = 0; i < this -> groupOffsets. getSize() - 1; i ++ )
   {
	   if( i * groupSize > print_lines )
		   return;
	   str << endl << "Block number: " << i << endl;
	   str << " Lines: " << i * groupSize << " -- " << ( i + 1 ) * groupSize << endl;
	   str << " Lines non-zeros: ";
	   for( Index k = i * groupSize; k < ( i + 1 ) * groupSize && k < this -> getSize(); k ++ )
		   str << nonzeroElementsInRow. getElement( k ) << "  ";
	   str << endl;
	   str << " Block data: "
	       << groupOffsets. getElement( i ) << " -- "
	       << groupOffsets. getElement( i + 1 ) << endl;
	   str << " Data:   ";
	   for( Index k = groupOffsets. getElement( i );
	        k < groupOffsets. getElement( i + 1 );
	        k ++ )
		   str << setprecision( 5 ) << setw( 8 )
		       << nonzeroElements. getElement( k ) << " ";
	   str << endl << "Columns: ";
	   for( Index k = groupOffsets. getElement( i );
	        k < groupOffsets. getElement( i + 1 );
	        k ++ )
		   str << setprecision( 5 ) << setw( 8 )
		       << columns. getElement( k ) << " ";
   }
   str << endl;
};

template< typename Real, tnlDevice Device, typename Index >
Index tnlRgCSRMatrix< Real, Device, Index > :: getCurrentGroupSize( const Index groupId ) const
{
   if( this -> useAdaptiveGroupSize )
   {
      tnlAssert( this -> adaptiveGroupSizes. getSize() > groupId,
                 cerr << "adaptiveGroupSizes. getSize() = " << adaptiveGroupSizes. getSize()
                      << " groupId = " << groupId );
      return this -> adaptiveGroupSizes[ groupId ];
   }
   else
   {
      /****
       * The last block may be smaller then the global groupSize.
       */
      if( ( groupId + 1 ) * this -> groupSize > this -> getSize() )
         return this -> getSize() % this -> groupSize;
      return this -> groupSize;
   }
}

#ifdef HAVE_CUDA
bool bindRgCSRMatrixCUDATexture( const float* data,
                                 size_t size )
{
#if CUDA_ARCH > 12
   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc< float >();
   size_t offset( -1 );
   cudaBindTexture( &offset,
                    &tnlRgCSRMatrixCUDA_floatTexRef,
                    data,
                    &channelDesc,
                    size * sizeof( float ) );
   if( ! checkCUDAError( __FILE__, __LINE__ ) )
      return false;
   if( offset != 0 )
   {
      std :: cerr << "Unable to bind the vector x into a texture!!!" << endl;
      return false;
   }
   return true;
#else
   return false;
#endif
};

bool bindRgCSRMatrixCUDATexture( const double* data,
                                 size_t size )
{
#if CUDA_ARCH > 12
   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc< int2 >();
   size_t offset( -1 );
   cudaBindTexture( &offset,
                    &tnlRgCSRMatrixCUDA_doubleTexRef,
                    data,
                    &channelDesc,
                    size * sizeof( double ) );
   if( ! checkCUDAError( __FILE__, __LINE__ ) )
      return false;
   if( offset != 0 )
   {
      std :: cerr << "Unable to bind the vector x into a texture!!!" << endl;
      return false;
   }
   return true;
#else
   return false;
#endif
};

template< bool UseCache, typename Index >
static __inline__ __device__ float fetchVecX( const Index i,
                                              const float* x )
{
    if( UseCache )
        return tex1Dfetch( tnlRgCSRMatrixCUDA_floatTexRef, i );
    else
        return x[ i ];
};

template< bool UseCache, typename Index >
static __inline__ __device__ double fetchVecX( const Index i,
                                               const double* x )
{
#if ( CUDA_ARCH >= 13 )
    if( UseCache )
    {
       int2 v=  tex1Dfetch( tnlRgCSRMatrixCUDA_doubleTexRef, i );
       return __hiloint2double(v.y, v.x);
    }
    else
        return x[ i ];
#endif
};

bool __inline__ unbindRgCSRMatrixCUDATexture( const float* dummy_pointer )
{
   cudaUnbindTexture( &tnlRgCSRMatrixCUDA_floatTexRef );
   return checkCUDAError( __FILE__, __LINE__ );
};

bool __inline__ unbindRgCSRMatrixCUDATexture( const double* dummy_pointer )
{
   cudaUnbindTexture( &tnlRgCSRMatrixCUDA_doubleTexRef );
   return checkCUDAError( __FILE__, __LINE__ );
};


template< typename Real, typename Index, bool useCache >
__global__ void sparseOldCSRMatrixVectorProductKernel( Index size,
                                                       Index block_size,
                                                       const Real* nonzeroElements,
                                                       const Index* columns,
                                                       const Index* groupOffsets,
                                                       const Index* nonzeroElementsInRow,
                                                       const Real* vec_x,
                                                       Real* vec_b )
{
   /****
    * Each thread process one matrix row
    */
   Index row = blockIdx. x * blockDim. x + threadIdx. x;
   if( row >= size )
      return;

   Index groupOffset = groupOffsets[ blockIdx. x ];
   Index pos = groupOffset + threadIdx. x;

   /****
    * The last block may be smaller then the global block_size.
    * We store it in the current_block_size
    */
   Index current_block_size = blockDim. x;
   if( ( blockIdx. x + 1 ) * blockDim. x > size )
      current_block_size = size % blockDim. x;

   Real product( 0.0 );
   const Index nonzeros = nonzeroElementsInRow[ row ];
   for( Index i = 0; i < nonzeros; i ++ )
   {
      //product += nonzeroElements[ pos ] * vec_x[ columns[ pos ] ];
      product += nonzeroElements[ pos ] * fetchVecX< useCache >( columns[ pos ], vec_x );
      pos += current_block_size;
   }
   vec_b[ row ] = product;
}


template< class Real, typename Index, bool useCache >
__global__ void sparseCSRMatrixVectorProductKernel( Index size,
                                                    Index groupSize,
                                                    const Real* nonzeroElements,
                                                    const Index* columns,
                                                    const Index* groupOffsets,
                                                    const Index* nonzerosInRow,
                                                    const Real* vec_x,
                                                    Real* vec_b )
{
   /****
    * Each thread process one matrix row
    */
   const Index rowIndex = blockIdx. x * blockDim. x + threadIdx. x;
   if( rowIndex >= size )
      return;

   const Index groupIndex = threadIdx . x / groupSize ;
   const Index rowOffsetInGroup = rowIndex % groupSize ;

   /****
    * The last block may be smaller then the global block_size.
    * We store it in the current_block_size
    */
   Index currentGroupSize = groupSize;
   if( ( blockIdx. x + 1 ) * blockDim. x > size )
      currentGroupSize =  size % groupSize;

   Real product( 0.0 );
   Index pos = groupOffsets[ rowIndex / groupSize ] + rowOffsetInGroup;
   const Index nonzeros = nonzerosInRow[ rowIndex ];
   for( Index i = 0; i < nonzeros; i ++ )
   {
      product += nonzeroElements[ pos ] * fetchVecX< useCache >( columns[ pos ], vec_x );
      pos += currentGroupSize;
   }
   vec_b[ rowIndex ] = product;
}

#endif // ifdef HAVE_CUDA


#endif /* TNLRgCSRMATRIX_H_ */