/***************************************************************************
                          tnlFastRgCSRMatrix.h  -  description
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

#ifndef TNLFastRgCSRMATRIXCUDA_H_
#define TNLFastRgCSRMATRIXCUDA_H_

#include <iostream>
#include <iomanip>
#include <core/tnlLongVector.h>
#include <core/tnlAssert.h>
#include <core/mfuncs.h>
#include <matrix/tnlMatrix.h>
#include <matrix/tnlFastRgCSRMatrix.h>
#include <debug/tnlDebug.h>

#ifdef HAVE_CUDA
void sparseFastCSRMatrixVectorProductKernelCaller( int size,
 												   int block_size,
												   const int* columns_sequences_blocks_offsets,
												   const int* columns_sequences_offsets,
												   const int* column_sequences,
												   const float* nonzero_elements,
												   const int* block_offsets,
												   const int* column_sequences_lengths,
												   const int* column_sequences_in_block,
												   const float* vec_x,
												   float* vec_b );

void sparseFastCSRMatrixVectorProductKernelCaller( int size,
										           int block_size,
												   const int* columns_sequences_blocks_offsets,
												   const int* columns_sequences_offsets,
												   const int* column_sequences,
												   const double* nonzero_elements,
												   const int* block_offsets,
												   const int* column_sequences_lengths,
												   const int* column_sequences_in_block,
												   const double* vec_x,
												   double* vec_b );
#endif


template< typename Real, typename Index >
class tnlFastRgCSRMatrix< Real, tnlCuda, Index > : public tnlMatrix< Real, tnlCuda, Index >
{
   public:
   //! Basic constructor
   tnlFastRgCSRMatrix( const char* name );

   const tnlString& getMatrixClass() const;

   tnlString getType() const;

   //! Sets the number of row and columns.
   bool setSize( int new_size );

   //! Allocate memory for the nonzero elements.
   bool setNonzeroElements( int elements );

   void reset();

   int getNonzeroElements() const;

   int getArtificialZeroElements() const;

   int getMaxColumnSequenceDictionarySize() const;

   bool setElement( int row,
                    int colum,
                    const Real& value )
   { abort(); };

   bool addToElement( int row,
                      int column,
                      const Real& value )
   { abort(); };

   bool copyFrom( const tnlFastRgCSRMatrix< Real, tnlHost >& csr_matrix );

   Real getElement( int row,
                    int column ) const;

   Real rowProduct( int row,
                    const tnlLongVector< Real, tnlCuda, Index >& vector ) const;

   void vectorProduct( const tnlLongVector< Real, tnlCuda, Index >& x,
                       tnlLongVector< Real, tnlCuda, Index >& b ) const;

   Real getRowL1Norm( int row ) const
   { abort(); };

   void multiplyRow( int row, const Real& value )
   { abort(); };

   //! Prints out the matrix structure
   void printOut( ostream& str,
		          const int lines = 0 ) const;

   protected:

   //! Insert one block to the matrix.
   /*! If there is some data already in this @param row it will be rewritten.
    *  @param elements says number of non-zero elements which will be inserted.
    *  @param data is pointer to the elements values.
    *  @param first_column is the column of the first non-zero element.
    *  @param offsets is a pointer to field with offsets of the elements with
    *  respect to the first one. All of them must sorted increasingly.
    *  The elements which do not fit to the matrix are omitted.
    */
   bool insertBlock( );

   //! This array stores the non-zero elements of the sparse matrix.
   tnlLongVector< Real, tnlCuda > nonzero_elements;

   tnlLongVector< int, tnlCuda, int > block_offsets;


   //! This array stores so called 'column sequences'.
   /*! In the common CSr format there is a sequence of column indices for each row telling
    *  in what column given non-zero element lives. We take this sequence and subtract row index
    *  from each index of this sequence. Thus we get relative offsets from the diagonal entry
    *  for each non-zero element. If the matrix is structured in some way these sequences might
    *  be the same for some rows.Then we do not need to store them all but just to refer several
    *  times the same sequence.
    *  This array is allocated by the same size as the @param nonzero_elements. However, not all
    *  alocated memory is used.
    */
   tnlLongVector< int, tnlCuda, int > column_sequences;

   //! This arrays stores the offsets of the column sequences begins in the column_sequences.
   /*! This array is allocated by the same size as the matrix size is. However, there might
    *  be less column sequences then the matrix rows.
    */
   tnlLongVector< int, tnlCuda, int > columns_sequences_offsets;

   //! This says where given block of column sequences begins
   tnlLongVector< int, tnlCuda, int > columns_sequences_blocks_offsets;

   tnlLongVector< int, tnlCuda, int > column_sequences_in_block;

   //! This array stores the lengths of each column sequence.
   /*! This array is allocated by the same size as the matrix size is. However, there might
    *  be less column sequences then the matrix rows.
    */
   tnlLongVector< int, tnlCuda, int > column_sequences_lengths;

   int block_size;

   int artificial_zeros;

   //! The last non-zero element is at the position last_non_zero_element - 1
   int last_nonzero_element;

   int column_sequences_length;

   int max_column_sequences_block_size;
};

template< typename Real, typename Index >
tnlFastRgCSRMatrix< Real, tnlCuda, Index > :: tnlFastRgCSRMatrix( const char* name )
   : tnlMatrix< Real, tnlCuda, Index >( name ),
     nonzero_elements( "tnlFastRgCSRMatrix< Real, tnlCuda, Index > :: nonzero-elements" ),
     block_offsets( "tnlFastRgCSRMatrix< Real, tnlCuda, Index > :: block-offsets" ),
     column_sequences( "tnlFastRgCSRMatrix< Real, tnlCuda, Index > :: column-sequences" ),
     columns_sequences_offsets( "tnlFastRgCSRMatrix< Real, tnlCuda, Index > :: columns-sequences-offsets" ),
     columns_sequences_blocks_offsets( "tnlFastRgCSRMatrix< Real, tnlCuda, Index > :: columns-sequences-blocks-offsets" ),
     column_sequences_in_block( "tnlFastRgCSRMatrix< Real, tnlCuda, Index > :: columns-sequences-in-block" ),
     column_sequences_lengths( "tnlFastRgCSRMatrix< Real, tnlCuda, Index > :: column-sequences-lengths" ),
     block_size( 0 ),
     artificial_zeros( 0 ),
     column_sequences_length( 0 )
{
};

template< typename Real, typename Index >
const tnlString& tnlFastRgCSRMatrix< Real, tnlCuda, Index > :: getMatrixClass() const
{
   return tnlMatrixClass :: main;
};

template< typename Real, typename Index >
tnlString tnlFastRgCSRMatrix< Real, tnlCuda, Index > :: getType() const
{
   return tnlString( "tnlFastRgCSRMatrix< ") + tnlString( GetParameterType( Real( 0.0 ) ) ) + tnlString( ", tnlCuda >" );
};

template< typename Real, typename Index >
bool tnlFastRgCSRMatrix< Real, tnlCuda, Index > :: setSize( int new_size )
{
   this -> size = new_size;
   int blocks_number = this -> size / block_size + ( this -> size % block_size != 0 );
   if( ! block_offsets. setSize( blocks_number + 1 ) ||
	   ! columns_sequences_offsets. setSize( this -> size + 1 ) ||
	   ! column_sequences_in_block. setSize( blocks_number ) ||
	   ! columns_sequences_blocks_offsets. setSize( blocks_number + 1 ) ||
	   ! column_sequences_lengths. setSize( this -> size ) )
      return false;
   block_offsets. setValue( 0 );
   columns_sequences_blocks_offsets. setValue( 0 );
   column_sequences_in_block. setValue( 0 );
   columns_sequences_offsets. setValue( 0 );
   column_sequences_lengths. setValue( 0 );
   last_nonzero_element = 0;
   return true;
};

template< typename Real, typename Index >
bool tnlFastRgCSRMatrix< Real, tnlCuda, Index > :: setNonzeroElements( int elements )
{
/*   if( ! nonzero_elements. setSize( elements ) )
      return false;
   nonzero_elements. setValue( 0.0 );
   return true;*/
   return false;
};

template< typename Real, typename Index >
void tnlFastRgCSRMatrix< Real, tnlCuda, Index > :: reset()
{
   nonzero_elements. reset();
   block_offsets. reset();
   column_sequences. reset();
   columns_sequences_offsets. reset();
   columns_sequences_blocks_offsets. reset();
   column_sequences_in_block. reset();
   column_sequences_lengths. reset();
   block_size = 0;
   artificial_zeros = 0;
   column_sequences_length = 0;
};

template< typename Real, typename Index >
int tnlFastRgCSRMatrix< Real, tnlCuda, Index > :: getNonzeroElements() const
{
   tnlAssert( nonzero_elements. getSize() > artificial_zeros, );
   return nonzero_elements. getSize() - artificial_zeros;
};

template< typename Real, typename Index >
int tnlFastRgCSRMatrix< Real, tnlCuda, Index > :: getArtificialZeroElements() const
{
	return artificial_zeros;
};

template< typename Real, typename Index >
int tnlFastRgCSRMatrix< Real, tnlCuda, Index > :: getMaxColumnSequenceDictionarySize() const
{
	return  max_column_sequences_block_size;
};


template< typename Real, typename Index >
bool tnlFastRgCSRMatrix< Real, tnlCuda, Index > :: copyFrom( const tnlFastRgCSRMatrix< Real, tnlHost >& coa_fast_csr_matrix )
{
	dbgFunctionName( "tnlFastRgCSRMatrix< Real, tnlCuda >", "copyFrom" );

	if( coa_fast_csr_matrix. getMaxColumnSequenceDictionarySize() > 10240 / 4 )
	{
		cerr << "ERROR: This matrix requires too large column sequences dictionary ( " << coa_fast_csr_matrix. getMaxColumnSequenceDictionarySize() << " )." << endl;
		return false;
	}
	block_size = coa_fast_csr_matrix. block_size;

	if( ! this -> setSize( coa_fast_csr_matrix. getSize() ) ||
    	! nonzero_elements. setSize( coa_fast_csr_matrix. nonzero_elements. getSize() ) ||
		! column_sequences. setSize( coa_fast_csr_matrix. column_sequences. getSize() ) )
		return false;

	nonzero_elements = coa_fast_csr_matrix. nonzero_elements;
	block_offsets = coa_fast_csr_matrix. block_offsets;
	column_sequences = coa_fast_csr_matrix. column_sequences;
	columns_sequences_offsets =  coa_fast_csr_matrix. columns_sequences_offsets;
	column_sequences_in_block =  coa_fast_csr_matrix. column_sequences_in_block;
	columns_sequences_blocks_offsets = coa_fast_csr_matrix. columns_sequences_blocks_offsets;
	column_sequences_lengths =  coa_fast_csr_matrix. column_sequences_lengths;

	/*{
		cerr << "Unable to transfer data from the host to the CUDA device." << endl;
		return false;
	}*/
	artificial_zeros = coa_fast_csr_matrix. getArtificialZeroElements();
	last_nonzero_element = coa_fast_csr_matrix. last_nonzero_element;
	column_sequences_length = coa_fast_csr_matrix. column_sequences_length;
	max_column_sequences_block_size = coa_fast_csr_matrix. max_column_sequences_block_size;

	return true;
};

template< typename Real, typename Index >
Real tnlFastRgCSRMatrix< Real, tnlCuda, Index > :: getElement( int row,
                                                        int column ) const
{
    tnlAssert( 0 <= row && row < this -> getSize(),
	 		   cerr << "The row is outside the matrix." );
    tnlAssert( 0 <= column && column < this -> getSize(),
	    	   cerr << "The column is outside the matrix." );
	 tnlAssert( false, cerr << "Not Implemeted Yet!" );
	 return 0;
}

template< typename Real, typename Index >
Real tnlFastRgCSRMatrix< Real, tnlCuda, Index > :: rowProduct( int row,
                                                               const tnlLongVector< Real, tnlCuda, Index >& vec ) const
{
   tnlAssert( 0 <= row && row < this -> getSize(),
			  cerr << "The row is outside the matrix." );
   tnlAssert( vec != NULL, );
   tnlAssert( false, cerr << "Not Implemented Yet!" );
   return 0;
}

template< typename Real, typename Index >
void tnlFastRgCSRMatrix< Real, tnlCuda, Index > :: vectorProduct( const tnlLongVector< Real, tnlCuda, Index >& vec,
                                                                  tnlLongVector< Real, tnlCuda, Index >& result ) const
{
   tnlAssert( vec != NULL && result != NULL,);

#ifdef HAVE_CUDA
	/*sparseFastCSRMatrixVectorProductKernel( this -> getSize(),
	                                              block_size,
	                                              columns_sequences_blocks_offsets. getVector(),
	                                              columns_sequences_offsets. getVector(),
	                                              column_sequences. getVector(),
	                                              nonzero_elements. getVector(),
	                                              block_offsets. getVector(),
	                                              column_sequences_lengths. getVector(),
	                                              column_sequences_in_block. getVector(),
	                                              vec,
	                                              result );*/
#else
	cerr << "CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif
};



template< typename Real, typename Index >
void tnlFastRgCSRMatrix< Real, tnlCuda, Index > :: printOut( ostream& str,
		                                                     const int lines ) const
{
   str << "Structure of tnlFastRgCSRMatrix" << endl;
   str << "Matrix name:" << this -> getName() << endl;
   str << "Matrix size:" << this -> getSize() << endl;
   str << "Allocated elements:" << nonzero_elements. getSize() << endl;
   str << "Matrix blocks: " << block_offsets. getSize() << endl;

   int print_lines = lines;
   if( ! print_lines )
	   print_lines = this -> getSize();

   for( int i = 0; i < this -> block_offsets. getSize() - 1; i ++ )
   {
	   if( i * block_size > print_lines )
		   continue;
	   str << endl << "Block number: " << i << endl;
	   str << " Lines: " << i * block_size << " -- " << ( i + 1 ) * block_size << endl;
	   str << " Column sequences: " << column_sequences_in_block. getElement( i ) << endl;
	   for( int k = i * block_size; k < ( i + 1 ) * block_size && k < this -> getSize(); k ++ )
	   {
		   str << " Line: " << k << flush
			   << " Line length: " << column_sequences_lengths. getElement( k ) << flush
			   << " Column sequence offset: " << columns_sequences_offsets. getElement( k ) << endl
			   << " Column sequence: ";
		   for( int l = 0; l < column_sequences_lengths. getElement( k ); l ++ )
		      str << column_sequences. getElement( columns_sequences_offsets. getElement( k ) + l * column_sequences_in_block. getElement( i ) ) + k << "  ";
		   str << endl;
	   }
	   str << endl;

	   int current_block_size = block_size;
	   if( ( i + 1 ) * block_size > this -> getSize() )
	      current_block_size = this -> getSize() % block_size;
	   int block_length = block_offsets. getElement( i + 1 ) - block_offsets. getElement( i );
	   int row_length = block_length / block_size;
	   str << " Block data: " << block_offsets. getElement( i ) << " -- " << block_offsets. getElement( i + 1 ) << endl;
	   str << " Block size: " << current_block_size << endl;
	   str << " Data:   " << endl;
	   for( int k = 0; k < current_block_size; k ++ )
	   {
	      str << " Block row " << k << " (" << i * block_size + k << ") : ";
	      for( int l = 0; l < row_length; l ++ )
	         str << setprecision( 5 ) << setw( 8 ) << nonzero_elements. getElement( block_offsets. getElement( i ) + l * current_block_size + k ) << " ";
	      str << endl;
	   }
   }
   str << endl;
   /*str << "*********************************************************" << endl;
   str << column_sequences << endl;*/
};

#ifdef HAVE_CUDA

template< typename Real, typename Index >
__global__ void sparseFastCSRMatrixVectorProductKernel( int size,
                                                      int block_size,
                                                      const int* columns_sequences_blocks_offsets,
                                                      const int* columns_sequences_offsets,
                                                      const int* column_sequences,
                                                      const Real* nonzero_elements,
                                                      const int* block_offsets,
                                                      const int* column_sequences_lengths,
                                                      const int* column_sequences_in_block,
                                                      const Real* vec_x,
                                                      Real* vec_b )
{
   __shared__ int columns[ 2560 ];
   /*
    * First copy the column sequence dictionary to the local memory.
    */
   const int directory_begin = columns_sequences_blocks_offsets[ blockIdx. x ];
   const int directory_size = columns_sequences_blocks_offsets[ blockIdx. x + 1 ] -
                            directory_begin;
   if( directory_size > 2560 )
      return;
   int i = threadIdx.x;
   while( i < directory_size )
   {
      columns[ i ] = column_sequences[ directory_begin + i ];
      i += blockDim. x;
   }

   const int column_seq_in_block = column_sequences_in_block[ blockIdx. x ];

   /*
    * Each thread process one matrix row
    */
   int row = blockIdx. x * blockDim. x + threadIdx. x;
   if( row >= size )
      return;

   int block_offset = block_offsets[ blockIdx. x ];
   int pos = block_offset + threadIdx. x;

   /*
    * The last block may be smaller then the global block_size.
    * We store it in the current_block_size
    */
   int current_block_size = blockDim. x;
   if( ( blockIdx. x + 1 ) * blockDim. x > size )
      current_block_size = size % blockDim. x;

   Real product( 0.0 );
   const int nonzeros = column_sequences_lengths[ row ];
   int columns_offset = columns_sequences_offsets[ row ] - directory_begin;
   for( int i = 0; i < nonzeros; i ++ )
   {
      product += nonzero_elements[ pos ] * vec_x[ columns[ columns_offset ] + row ];
      pos += current_block_size;
       columns_offset += column_seq_in_block;
   }
   vec_b[ row ] = product;
}

#endif // ifdef HAVE_CUDA

#endif /* TNLFastRgCSRMATRIXCUDA_H_ */
