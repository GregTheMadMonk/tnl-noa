/***************************************************************************
                          tnlFastRgCSRMatrix.h  -  description
                             -------------------
    begin                : Jul 10, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLFastRgCSRMATRIX_H_
#define TNLFastRgCSRMATRIX_H_

#include <iostream>
#include <iomanip>
#include <core/vectors/tnlVector.h>
#include <core/tnlAssert.h>
#include <core/mfuncs.h>
#include <matrices/tnlMatrix.h>
#include <debug/tnlDebug.h>

//! Matrix storing the non-zero elements in the Row-Grouped CSR (Compressed Sparse Row) format
/*!
 */
template< typename Real, typename device = tnlHost, typename Index = int >
class tnlFastRgCSRMatrix
{
};

template< typename Real, typename Index >
class tnlFastRgCSRMatrix< Real, tnlHost, Index > : public tnlMatrix< Real, tnlHost, Index >
{
   public:
   //! Basic constructor
   tnlFastRgCSRMatrix( const tnlString& name, Index _block_size );

   const tnlString& getMatrixClass() const;

   tnlString getType() const;

   //! Sets the number of row and columns.
   bool setSize( Index new_size );

   //! Allocate memory for the nonzero elements.
   bool setNonzeroElements( Index elements );

   void reset();

   Index getNonzeroElements() const;

   Index getArtificialZeroElements() const;

   Index getMaxColumnSequenceDictionarySize() const;

   bool setElement( Index row,
                    Index colum,
                    const Real& value )
   { abort(); };

   bool addToElement( Index row,
                      Index column,
                      const Real& value )
   { abort(); };

   bool copyFrom( const tnlFastCSRMatrix< Real, tnlHost >& csr_matrix );

   Real getElement( Index row,
                    Index column ) const;

   Real rowProduct( Index row,
                    const tnlVector< Real, tnlHost, Index >& vector ) const;

   void vectorProduct( const tnlVector< Real, tnlHost, Index >& x,
                       tnlVector< Real, tnlHost, Index >& b ) const;

   Real getRowL1Norm( Index row ) const
   { abort(); };

   void multiplyRow( Index row, const Real& value )
   { abort(); };

   //! Prints out the matrix structure
   void printOut( ostream& str,
		          const Index lines = 0 ) const;

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
   tnlVector< Real > nonzero_elements;

   tnlVector< Index > block_offsets;


   //! This array stores so called 'column sequences'.
   /*! In the common CSr format there is a sequence of column indices for each row telling
    *  in what column given non-zero element lives. We take this sequence and subtract row index
    *  from each index of this sequence. Thus we get relative offsets from the diagonal entry
    *  for each non-zero element. If the matrix is structured in some way these sequences might
    *  be the same for some rows.Then we do not need to store them all but just to refer several
    *  times the same sequence.
    *  This array is allocated by the same size as the @param nonzero_elements. However, not all
    *  allocated memory is used.
    */
   tnlVector< Index > column_sequences;

   //! This arrays stores the offsets of the column sequences begins in the column_sequences.
   /*! This array is allocated by the same size as the matrix size is. However, there might
    *  be less column sequences then the matrix rows.
    */
   tnlVector< Index > columns_sequences_offsets;

   //! This says where given block of column sequences begins
   tnlVector< Index > columns_sequences_blocks_offsets;

   tnlVector< Index > column_sequences_in_block;

   //! This array stores the lengths of each column sequence.
   /*! This array is allocated by the same size as the matrix size is. However, there might
    *  be less column sequences then the matrix rows.
    */
   tnlVector< Index > column_sequences_lengths;

   Index block_size;

   Index artificial_zeros;

   //! The last non-zero element is at the position last_non_zero_element - 1
   Index last_nonzero_element;

   Index column_sequences_length;

   Index max_column_sequences_block_size;

   friend class tnlFastRgCSRMatrix< Real, tnlCuda >;
};

template< typename Real, typename Index >
tnlFastRgCSRMatrix< Real, tnlHost, Index > :: tnlFastRgCSRMatrix( const tnlString& name, Index _block_size )
   : tnlMatrix< Real, tnlHost, Index >( name ),
     nonzero_elements( "tnlFastRgCSRMatrix< Real, tnlHost, Index > :: nonzero-elements" ),
     block_offsets( "tnlFastRgCSRMatrix< Real, tnlHost, Index > :: block-offsets" ),
     column_sequences( "tnlFastRgCSRMatrix< Real, tnlHost, Index > :: column-sequences" ),
     columns_sequences_offsets( "tnlFastRgCSRMatrix< Real, tnlHost, Index > :: columns-sequences-offsets" ),
     columns_sequences_blocks_offsets( "tnlFastRgCSRMatrix< Real, tnlHost, Index > :: columns-sequences-blocks-offsets" ),
     column_sequences_in_block( "tnlFastRgCSRMatrix< Real, tnlHost, Index > :: columns-sequences-in-block" ),
     column_sequences_lengths( "tnlFastRgCSRMatrix< Real, tnlHost, Index > :: column-sequences-lengths" ),
     block_size( _block_size ),
     artificial_zeros( 0 ),
     column_sequences_length( 0 )
{
};

template< typename Real, typename Index >
const tnlString& tnlFastRgCSRMatrix< Real, tnlHost, Index > :: getMatrixClass() const
{
   return tnlMatrixClass :: main;
};

template< typename Real, typename Index >
tnlString tnlFastRgCSRMatrix< Real, tnlHost, Index > :: getType() const
{
   return tnlString( "tnlFastRgCSRMatrix< ") + tnlString( getType( Real( 0.0 ) ) ) + tnlString( ", tnlHost >" );
};

template< typename Real, typename Index >
bool tnlFastRgCSRMatrix< Real, tnlHost, Index > :: setSize( Index new_size )
{
   this->size = new_size;
   Index blocks_number = this->size / block_size + ( this->size % block_size != 0 );
   if( ! block_offsets. setSize( blocks_number + 1 ) ||
	   ! columns_sequences_blocks_offsets. setSize( blocks_number + 1 ) ||
	   ! column_sequences_in_block. setSize( blocks_number ) ||
	   ! columns_sequences_offsets. setSize( this->size + 1 ) ||
	   ! column_sequences_lengths. setSize( this->size ) )
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
bool tnlFastRgCSRMatrix< Real, tnlHost, Index > :: setNonzeroElements( Index elements )
{
   if( ! nonzero_elements. setSize( elements ) )
      return false;
   nonzero_elements. setValue( 0.0 );
   return true;
};

template< typename Real, typename Index >
void tnlFastRgCSRMatrix< Real, tnlHost, Index > :: reset()
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
}

template< typename Real, typename Index >
Index tnlFastRgCSRMatrix< Real, tnlHost, Index > :: getNonzeroElements() const
{
   tnlAssert( nonzero_elements. getSize() > artificial_zeros, );
   return nonzero_elements. getSize() - artificial_zeros;
};

template< typename Real, typename Index >
Index tnlFastRgCSRMatrix< Real, tnlHost, Index > :: getArtificialZeroElements() const
{
	return artificial_zeros;
};

template< typename Real, typename Index >
Index tnlFastRgCSRMatrix< Real, tnlHost, Index > :: getMaxColumnSequenceDictionarySize() const
{
	return  max_column_sequences_block_size;
};


template< typename Real, typename Index >
bool tnlFastRgCSRMatrix< Real, tnlHost, Index > :: copyFrom( const tnlFastCSRMatrix< Real, tnlHost >& fast_csr_matrix )
{
	dbgFunctionName( "tnlFastRgCSRMatrix< Real, tnlHost >", "copyFrom" );
	if( ! this->setSize( fast_csr_matrix. getSize() ) )
		return false;

	Index blocks_number = this->size / block_size + ( this->size % block_size != 0 );
	tnlVector< Index > col_seq_block_size( "tnlFastRgCSRMatrix< Real, tnlHost, Index > :: col_seq_block_size" );
	col_seq_block_size. setSize( blocks_number );
	col_seq_block_size. setValue( 0 );
	column_sequences_lengths. setSize( fast_csr_matrix. column_sequences_lengths. getSize() );
	column_sequences_lengths = fast_csr_matrix. column_sequences_lengths;
	max_column_sequences_block_size = 0;
	Index sequences_in_block( 0 );
	Index longest_sequence_length( 0 );
	/*
	 *	First compute the column sequences block sizes.
	 *	The sequences must be also stored in the coalsced-like style.
	 *	Therefore the column sequences block size is given as the length
	 *	of the longest column sequences times number of sequences.
	 *	The length of the longest sequence is stored in longest_sequence_length,
	 *	and the number of sequences in one block is stored in columns_sequences_in_block
	 */
	for( Index i = 0; i < this->getSize(); i ++ )
	{
		Index block_id = i / block_size;
		/*
		 * The last block may be smaller then the global block_size.
		 * We store it in the current_block_size
		 */
		Index current_block_size = block_size;
		if( ( block_id + 1 ) * block_size > this->getSize() )
		   current_block_size = this->getSize() % block_size;

		Index current_column_sequence = fast_csr_matrix. columns_sequences_offsets[ i ];

		/*
		 * Check if this sequence was already used in this block.
		 * We look for a row where there is the same column sequence.
		 * This can be done in the fast_csr_matrix because it is easier.
		 * It is not enough to compare the column-sequence offsets.
		 * In the case when we have empty row then the column-sequence have
		 * zero length and the next column-sequence starts at the same offset.
		 * Therefore we must also compare the column-sequence lengths.
		 */
		bool sequence_used( false );
		Index j = block_id * block_size;
		while( j < i && ! sequence_used )
		{
			if( fast_csr_matrix. columns_sequences_offsets[ j ] == fast_csr_matrix. columns_sequences_offsets[ i ] &&
			    fast_csr_matrix. column_sequences_lengths[ j ] == fast_csr_matrix. column_sequences_lengths[ i ] )
				sequence_used = true;
			j ++;
		}
		if( ! sequence_used )
		{
		   longest_sequence_length = Max( column_sequences_lengths[ i ], longest_sequence_length );
		   dbgCout( "Counting new column-sequence at line " << i );
		   column_sequences_in_block[ block_id ] ++;
		}
		if( i % block_size == current_block_size - 1 )
		{
		   dbgCout( "Block ID: " << block_id
		          << " Lines: " << block_size * block_id << "--" << block_size * block_id + current_block_size
				    << " Sequences in block: " << column_sequences_in_block[ block_id ]
				    << " Longest sequence: " << longest_sequence_length );
		   col_seq_block_size[ block_id ] = column_sequences_in_block[ block_id ] * longest_sequence_length;
		   longest_sequence_length = 0;
		   max_column_sequences_block_size = Max( max_column_sequences_block_size, col_seq_block_size[ block_id ] );
		}
	}

	/*if( max_column_sequences_block_size * sizeof( Index ) > 10240 )
	{
		cerr << "ERROR: This matrix requires too large column sequences dictionary ( " << max_column_sequences_block_size << " )." << endl;
		return false;
	}*/

	/*
	 * Now set columns sequences blocks offsets
	 */
	columns_sequences_blocks_offsets[ 0 ] = 0;
	for( Index i = 0; i < blocks_number; i ++ )
	{
		columns_sequences_blocks_offsets[ i + 1 ] = columns_sequences_blocks_offsets[ i ] + col_seq_block_size[ i ];
		dbgExpr( columns_sequences_blocks_offsets[ i + 1 ] );
	}

	/*
	 * Copy the column sequences and proper columns sequences offsets with respect
	 * to the new columns sequences blocks.
	 */
    column_sequences. setSize( columns_sequences_blocks_offsets[ blocks_number ] );
    column_sequences. setValue( -1 );
	Index column_sequences_end( 0 );
	Index inserted_column_sequences( 0 );
	for( Index i = 0; i < this->getSize(); i ++ )
	{
	   dbgCout( "Processing column-sequence for the line " << i );
		Index block_id = i / block_size;
		Index current_column_sequence = fast_csr_matrix. columns_sequences_offsets[ i ];

		/*
		 * Check if this sequence was already used in this block.
		 * We look for a row where there is the same column sequence.
		 * This can be done in the fast_csr_matrix because it is easier.
		 * It is not enough to compare the column-sequence offsets.
		 * In the case when we have empty row then the column-sequence have
		 * zero length and the next column-sequence starts at the same offset.
		 * Therefore we must also compare the column-sequence lengths.
		 */
		bool sequence_used( false );
		Index j = block_id * block_size;
		while( j < i && ! sequence_used )
		{
			if( fast_csr_matrix. columns_sequences_offsets[ j ] == fast_csr_matrix. columns_sequences_offsets[ i ] &&
			    fast_csr_matrix. column_sequences_lengths[ j ] == fast_csr_matrix. column_sequences_lengths[ i ] )
				sequence_used = true;
			j ++;
		}
		if( sequence_used )
		{
		   dbgCout( "Match found at column-sequence " << j - 1 );
			columns_sequences_offsets[ i ] = columns_sequences_offsets[ j - 1 ];
		}
		else
		{
			/*
			 * Copy the column sequence into this block of the column sequences
			 */
		   columns_sequences_offsets[ i ] = columns_sequences_blocks_offsets[ block_id ] + inserted_column_sequences;
			Index fast_csr_column_sequence_offset = fast_csr_matrix. columns_sequences_offsets[ i ];
			dbgCout( "Copying " << inserted_column_sequences << ". column sequence with the length "
			         << column_sequences_lengths[ i ] << " from the fast CSR matrix column offset " << fast_csr_matrix. columns_sequences_offsets[ i ] );
         tnlAssert( inserted_column_sequences < column_sequences_in_block[ block_id ],
                    cerr << "inserted_column_sequences = " << inserted_column_sequences << endl
                         << "column_sequences_in_block[ block_id ] = " << column_sequences_in_block[ block_id ] << endl
                         << "block_id = " << block_id );
			for( Index j = 0; j < column_sequences_lengths[ i ]; j ++ )
			{
			   tnlAssert( columns_sequences_offsets[ i ] + j * column_sequences_in_block[ block_id ] < columns_sequences_blocks_offsets[ block_id + 1],
			              cerr << "j = " << j << endl
			                   << "columns_sequences_offsets[ i ] = " << columns_sequences_offsets[ i ] << endl
			                   << "column_sequences_in_block[ block_id ] = " << column_sequences_in_block[ block_id ] << endl
			                   << "columns_sequences_offsets[ i ] + j * column_sequences_in_block[ block_id ] = " << columns_sequences_offsets[ i ] + j * column_sequences_in_block[ block_id ] << endl
			                   << "block_id = " << block_id << endl
			                   << "columns_sequences_blocks_offsets[ block_id ] = " << columns_sequences_blocks_offsets[ block_id ] << endl
			                   << "inserted_column_sequences = " << inserted_column_sequences << endl
			                   << "columns_sequences_blocks_offsets[ block_id + 1] = " << columns_sequences_blocks_offsets[ block_id + 1] );
				column_sequences[ columns_sequences_offsets[ i ] + j * column_sequences_in_block[ block_id ] ] = fast_csr_matrix. column_sequences[ fast_csr_column_sequence_offset + j ];
				//dbgExpr( fast_csr_matrix. column_sequences[ fast_csr_column_sequence_offset + j ] + i )
			}
			inserted_column_sequences ++;
		}
      if( i % block_size == block_size - 1 )
           inserted_column_sequences = 0;
	}

	/*
	 * Now we need to copy the nonzero_elements and the block offsets.
	 * We do it the same way as in the Coalesced CSR matrix.
	 * Firstly we compute the number of non-zero elements in each row
	 * and compute number of elements which are necessary allocate.
     */
	Index total_elements( 0 );
	Index max_row_in_block( 0 );
	Index blocks_inserted( -1 );
	for( Index i = 0; i < this->getSize(); i ++ )
	{
		if( i % block_size == 0 )
		{
			total_elements += max_row_in_block * block_size;
			block_offsets[ i / block_size ] = total_elements;
			blocks_inserted ++;
			//dbgExpr( block_offsets[ i / block_size ] );
			max_row_in_block = 0;
		}
		//nonzeros_in_row[ i ] = fast_csr_matrix. row_offsets[ i + 1 ] - fast_csr_matrix. row_offsets[ i ];
		//dbgExpr( nonzeros_in_row[ i ] );
		max_row_in_block = Max( max_row_in_block, column_sequences_lengths[ i ] );
	}
	total_elements += max_row_in_block * ( this->getSize() - blocks_inserted * block_size );
	block_offsets[ block_offsets. getSize() - 1 ] = total_elements;


	/*
	 * Allocate the non-zero elements (they contains some artificial zeros.)
	 */
	dbgExpr( total_elements );
	dbgCout( "Allocating " << total_elements << " elements.");
	if( ! setNonzeroElements( total_elements ) )
		return false;
	artificial_zeros = total_elements - fast_csr_matrix. getNonzeroElements();


	dbgCout( "Inserting data " );
	/*
	 * Insert the data into the blocks.
	 * We go through the blocks.
	 */
	for( Index i = 0; i < block_offsets. getSize() - 1; i ++ )
	{
		//dbgExpr( block_offsets[ i ] );
		/*
		 * The last block may be smaller then the global block_size.
		 * We store it in the current_block_size
		 */
		Index current_block_size = block_size;
		if( ( i + 1 ) * block_size > this->getSize() )
			current_block_size = this->getSize() % block_size;

		/*
		 * We insert 'current_block_size' rows in this matrix with the stride
		 * given by the block size.
		 */
		for( Index k = 0; k < current_block_size; k ++ )
		{
			/*
			 * We start with the offset k within the block and
			 * we insert the data with a stride equal to the block size.
			 * j - is the element position in the nonzero_elements in this matrix
			 */
			Index j = block_offsets[ i ] + k;                   // position of the first element of the row
			Index element_row = i * block_size + k;
			//dbgExpr( element_row );
			if( element_row < this->getSize() )
			{

				/*
				 * Get the element position
				 */
				Index element_pos = fast_csr_matrix. row_offsets[ element_row ];
				while( element_pos < fast_csr_matrix. row_offsets[ element_row + 1 ] )
				{
					/*dbgCout( "Inserting on position " << j
							 << " data " << csr_matrix. nonzero_elements[ element_pos ]
							 << " at column " << csr_matrix. columns[ element_pos ] );*/
					nonzero_elements[ j ] = fast_csr_matrix. nonzero_elements[ element_pos ];
					//columns[ j ] = fast_csr_matrix. columns[ element_pos ];

					element_pos ++;
					j += current_block_size;
				}
			}
		}
	}
	return true;

};

template< typename Real, typename Index >
Real tnlFastRgCSRMatrix< Real, tnlHost, Index > :: getElement( Index row,
                                                               Index column ) const
{
   tnlAssert( 0 <= row && row < this->getSize(),
			  cerr << "The row is outside the matrix." );
   tnlAssert( 0 <= column && column < this->getSize(),
			  cerr << "The column is outside the matrix." );

	Index block_id = row / block_size;
	Index block_row = row % block_size;
	Index block_offset = block_offsets[ block_id ];
	/*
	 * The last block may be smaller then the global block_size.
	 * We store it in the current_block_size
	 */
	Index current_block_size = block_size;
	if( ( block_id + 1 ) * block_size > this->getSize() )
		current_block_size = this->getSize() % block_size;
	Index pos = block_offset + block_row;

   Index column_offset = columns_sequences_offsets[ row ];
   const Index* cols = column_sequences. getData();
   Index row_length = column_sequences_lengths[ row ];
   Index columns_in_block = column_sequences_in_block[ block_id ];

   Index i = 0;
   while( i < row_length && cols[ column_offset + i * columns_in_block ] + row < column ) i ++;

   const Real* els = nonzero_elements. getData();
   if( i < row_length && cols[ column_offset + i * columns_in_block ] + row == column )
	  return els[ block_offset + block_row + i * current_block_size ];
   return Real( 0.0 );
}

template< typename Real, typename Index >
Real tnlFastRgCSRMatrix< Real, tnlHost, Index > :: rowProduct( Index row,
                                                               const tnlVector< Real, tnlHost, Index >& vec ) const
{
   tnlAssert( 0 <= row && row < this->getSize(),
           cerr << "The row is outside the matrix." );
   tnlAssert( vec. getSize() == this->getSize(),
              cerr << "The matrix and vector for multiplication have different sizes. "
                   << "The matrix size is " << this->getSize() << "."
                   << "The vector size is " << vec. getSize() << endl; );

   Index block_id = row / block_size;
   Index block_row = row % block_size;
   Index block_offset = block_offsets[ block_id ];
   /*
	* The last block may be smaller then the global block_size.
	* We store it in the current_block_size
	*/
   Index current_block_size = block_size;
   if( ( block_id + 1 ) * block_size > this->getSize() )
	   current_block_size = this->getSize() % block_size;
   Real product( 0.0 );
   Index val_pos = block_offset + block_row;
   Index column_pos = columns_sequences_offsets[ row ];
   const Index col_sequences_in_block = column_sequences_in_block[ block_id ];
   for( Index i = 0; i < column_sequences_lengths[ row ]; i ++ )
   {
	   tnlAssert( val_pos < nonzero_elements. getSize(), );
	   product += nonzero_elements[ val_pos ] * vec[ column_sequences[ column_pos ] + row ];
	   val_pos += current_block_size;
	   column_pos += col_sequences_in_block;
   }
   return product;
}

template< typename Real, typename Index >
void tnlFastRgCSRMatrix< Real, tnlHost, Index > :: vectorProduct( const tnlVector< Real, tnlHost, Index >& vec,
                                                                  tnlVector< Real, tnlHost, Index >& result ) const
{
   tnlAssert( vec. getSize() == this->getSize(),
              cerr << "The matrix and vector for a multiplication have different sizes. "
                   << "The matrix size is " << this->getSize() << "."
                   << "The vector size is " << vec. getSize() << endl; );
   tnlAssert( result. getSize() == this->getSize(),
              cerr << "The matrix and result vector of a multiplication have different sizes. "
                   << "The matrix size is " << this->getSize() << "."
                   << "The vector size is " << result. getSize() << endl; );

   for( Index row = 0; row < this->getSize(); row ++ )
   {
	   Index block_id = row / block_size;
	   Index block_row = row % block_size;
	   Index block_offset = block_offsets[ block_id ];
	   /*
		* The last block may be smaller then the global block_size.
		* We store it in the current_block_size
		*/
	   Index current_block_size = block_size;
	   if( ( block_id + 1 ) * block_size > this->getSize() )
		   current_block_size = this->getSize() % block_size;
	   Real product( 0.0 );
	   Index val_pos = block_offset + block_row;
	   Index column_pos = columns_sequences_offsets[ row ];
	   const Index col_sequences_in_block = column_sequences_in_block[ block_id ];
	   for( Index i = 0; i < column_sequences_lengths[ row ]; i ++ )
	   {
		   tnlAssert( val_pos < nonzero_elements. getSize(), );
		   product += nonzero_elements[ val_pos ] * vec[ column_sequences[ column_pos ] + row ];
		   val_pos += current_block_size;
		   column_pos += col_sequences_in_block;
	   }
	   result[ row ] = product;
   }
};



template< typename Real, typename Index >
void tnlFastRgCSRMatrix< Real, tnlHost, Index > :: printOut( ostream& str,
                                                             const tnlString& name,
		                                                       const Index lines ) const
{
   str << "Structure of tnlFastRgCSRMatrix" << endl;
   str << "Matrix name:" << name << endl;
   str << "Matrix size:" << this->getSize() << endl;
   str << "Allocated elements:" << nonzero_elements. getSize() << endl;
   str << "Matrix blocks: " << block_offsets. getSize() << endl;

   Index print_lines = lines;
   if( ! print_lines )
	   print_lines = this->getSize();

   for( Index i = 0; i < this->block_offsets. getSize() - 1; i ++ )
   {
	   if( i * block_size > print_lines )
		   continue;
	   str << endl << "Block number: " << i << endl;
	   str << " Lines: " << i * block_size << " -- " << ( i + 1 ) * block_size << endl;
	   str << " Column sequences: " << column_sequences_in_block[ i ] << endl;
	   for( Index k = i * block_size; k < ( i + 1 ) * block_size && k < this->getSize(); k ++ )
	   {
		   str << " Line: " << k << flush
			   << " Line length: " << column_sequences_lengths[ k ] << flush
			   << " Column sequence offset: " << columns_sequences_offsets[ k ] << endl
			   << " Column sequence: " << flush;
		   for( Index l = 0; l < column_sequences_lengths[ k ]; l ++ )
		      str << column_sequences[ columns_sequences_offsets[ k ] + l * column_sequences_in_block[ i ] ] + k << "  ";
		   str << endl;
	   }
	   str << endl;

	   Index current_block_size = block_size;
	   if( ( i + 1 ) * block_size > this->getSize() )
	      current_block_size = this->getSize() % block_size;
	   Index block_length = block_offsets[ i + 1 ] - block_offsets[ i ];
	   Index row_length = block_length / block_size;
	   str << " Block data: " << block_offsets[ i ] << " -- " << block_offsets[ i + 1 ] << endl;
	   str << " Block size: " << current_block_size << endl;
	   str << " Data:   " << endl;
	   for( Index k = 0; k < current_block_size; k ++ )
	   {
	      str << " Block row " << k << " (" << i * block_size + k << ") : ";
	      for( Index l = 0; l < row_length; l ++ )
	         str << setprecision( 5 ) << setw( 8 ) << nonzero_elements[ block_offsets[ i ] + l * current_block_size + k ] << " ";
	      str << endl;
	   }
   }
   str << endl;
   /*str << "*********************************************************" << endl;
   str << column_sequences << endl;*/
};




#endif /* TNLFastRgCSRMATRIX_H_ */
