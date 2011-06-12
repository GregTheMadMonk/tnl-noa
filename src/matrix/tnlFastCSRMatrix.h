/***************************************************************************
                          tnlFastCSRMatrix.h  -  description
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

#ifndef TNLFASTCSRMATRIX_H_
#define TNLFASTCSRMATRIX_H_

#include <iostream>
#include <iomanip>
#include <core/tnlLongVectorHost.h>
#include <core/tnlAssert.h>
#include <core/mfuncs.h>
#include <matrix/tnlMatrix.h>
#include <debug/tnlDebug.h>


using namespace std;

template< typename Real, tnlDevice Device, typename Index > class tnlFastRgCSRMatrix;


//! The Fast CSR format is based on the common CSR format but with compression of the column indices.
/*! The non-zero elements are stored in the same way as in the common CSR @see nonzero_elements
 *
    \author Tomas Oberhuber.
 */

template< typename Real, tnlDevice Device = tnlHost, typename Index = int >
class tnlFastCSRMatrix
{
};

template< typename Real, typename Index >
class tnlFastCSRMatrix< Real, tnlHost, Index > : public tnlMatrix< Real, tnlHost, Index >
{
   public:

   //! Basic constructor
   tnlFastCSRMatrix( const char* name );

   const tnlString& getMatrixClass() const;

   tnlString getType() const;

   //! Sets the number of row and columns.
   bool setSize( Index new_size );

   //! Allocate memory for the nonzero elements.
   bool setNonzeroElements( Index elements );

   void reset();

   Index getNonzeroElements() const;

   //! Return the lengths of the column sequences dictionary.
   /* It is only important for the format efficiency estimation.
    *
    */
   Index getColumnSequencesLength() const;


   bool insertRow( Index row,
                   Index elements,
                   Real* data,
                   Index first_column,
                   Index* offsets );

   bool setElement( Index row,
                    Index colum,
                    const Real& value );

   bool addToElement( Index row,
                      Index column,
                      const Real& value );

   bool copyFrom( const tnlCSRMatrix< Real, tnlHost, Index >& csr_matrix );

   Real getElement( Index row,
                    Index column ) const;

   Real rowProduct( Index row,
                    const tnlLongVector< Real, tnlHost, Index >& vector ) const;

   void vectorProduct( const tnlLongVector< Real, tnlHost, Index >& x,
                       tnlLongVector< Real, tnlHost, Index >& b ) const;

   Real getRowL1Norm( Index row ) const;

   void multiplyRow( Index row, const Real& value );

   //! Method for saving the matrix to a file as a binary data
   bool Save( ostream& file ) const;

   //! Method for restoring the matrix from a file
   bool Load( istream& file );

   //! Prints out the matrix structure
   void printOut( ostream& str,
		          const Index lines = 0 ) const;

   protected:

   enum elementOperation { set_element, add_to_element };

   Index getRowLength( Index row ) const;

   //! Returns the position of the element in the vector of the non-zero elements.
   /*!
    * If there is no such element the returned value points to the next element with
    * higher column index.
    */
   //Index getElementPosition( Index row,
   //                        Index colum ) const;

   Index insertColumnSequence( const tnlLongVector< Index >& columns,
	                         Index column_sequence_offset,
		                     Index column_sequence_length,
		                     Index row );


   //! This array stores the non-zero elements of the sparse matrix.
   tnlLongVector< Real > nonzero_elements;

   //! This array stores the indices of the row begins in the nonzero_elements array.
   /*! In the case of the common CSR format this array would be shared even with the column
    *  indices array but it is not the case now.
    */
   tnlLongVector< Index > row_offsets;

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
   tnlLongVector< Index > column_sequences;

   //! This arrays stores the offsets of the column sequences begins in the column_sequences.
   /*! This array is allocated by the same size as the matrix size is. However, there might
    *  be less column sequences then the matrix rows.
    */
   tnlLongVector< Index > columns_sequences_offsets;

   //! This array stores the lengths of each column sequence.
   /*! This array is allocated by the same size as the matrix size is. However, there might
    *  be less column sequences then the matrix rows.
    */
   tnlLongVector< Index > column_sequences_lengths;



   //! The last non-zero element is at the position last_non_zero_element - 1
   Index last_nonzero_element;

   Index column_sequences_length;

   friend class tnlFastRgCSRMatrix< Real, tnlHost, Index >;
   //friend class tnlEllpackMatrix< Real, tnlHost >;
};

template< typename Real, typename Index >
tnlFastCSRMatrix< Real, tnlHost, Index > :: tnlFastCSRMatrix( const char* name )
   : tnlMatrix< Real, tnlHost, Index >( name ),
     nonzero_elements( "tnlFastCSRMatrix< Real, tnlHost, Index > :: nonzero-elements" ),
     row_offsets( "tnlFastCSRMatrix< Real, tnlHost, Index > :: row-offsets" ),
     column_sequences( "tnlFastCSRMatrix< Real, tnlHost, Index > :: column-sequences" ),
     columns_sequences_offsets( "tnlFastCSRMatrix< Real, tnlHost, Index > :: columns-sequences-offsets" ),
     column_sequences_lengths( "tnlFastCSRMatrix< Real, tnlHost, Index > :: column-sequences-lengths" ),
     last_nonzero_element( 0 ),
     column_sequences_length( 0 )
{
};

template< typename Real, typename Index >
const tnlString& tnlFastCSRMatrix< Real, tnlHost, Index > :: getMatrixClass() const
{
   return tnlMatrixClass :: main;
};

template< typename Real, typename Index >
tnlString tnlFastCSRMatrix< Real, tnlHost, Index > :: getType() const
{
   return tnlString( "tnlFastCSRMatrix< ") + tnlString( GetParameterType( Real( 0.0 ) ) ) + tnlString( ", tnlHost >" );
};

template< typename Real, typename Index >
bool tnlFastCSRMatrix< Real, tnlHost, Index > :: setSize( Index new_size )
{
   this -> size = new_size;
   if( ! row_offsets. setSize( this -> size + 1 ) ||
	   ! columns_sequences_offsets. setSize( this -> size + 1 ) ||
	   ! column_sequences_lengths. setSize( this -> size ) )
      return false;
   row_offsets. setValue( 0 );
   columns_sequences_offsets. setValue( 0.0 );
   last_nonzero_element = 0;
   return true;
};

template< typename Real, typename Index >
bool tnlFastCSRMatrix< Real, tnlHost, Index > :: setNonzeroElements( Index elements )
{
   dbgFunctionName( "tnlFastCSRMatrix< Real, tnlHost >", "setNonzeroElements" );
   dbgExpr( elements );
   if( ! nonzero_elements. setSize( elements ) )
      return false;
   nonzero_elements. setValue( 0 );
   if( ! column_sequences. setSize( elements ) )
      return false;
   column_sequences. setValue( -1 );
   column_sequences_length = 0;
   return true;
};

template< typename Real, typename Index >
void tnlFastCSRMatrix< Real, tnlHost, Index > :: reset()
{
   nonzero_elements. reset();
   row_offsets. reset();
   column_sequences. reset();
   columns_sequences_offsets. reset();
   column_sequences_lengths. reset();
   last_nonzero_element = 0;
   column_sequences_length = 0;
}

template< typename Real, typename Index >
Index tnlFastCSRMatrix< Real, tnlHost, Index > :: getNonzeroElements() const
{
	return last_nonzero_element;
}

template< typename Real, typename Index >
Index tnlFastCSRMatrix< Real, tnlHost, Index > :: getColumnSequencesLength() const
{
	return column_sequences_length;
}

template< typename Real, typename Index >
Index tnlFastCSRMatrix< Real, tnlHost, Index > :: getRowLength( Index row ) const
{
	tnlAssert( row >= 0 && row < this -> getSize(), );
	return row_offsets[ row + 1 ] - row_offsets[ row ];
}

/*template< typename Real, typename Index >
Index tnlFastCSRMatrix< Real, tnlHost, Index > :: getElementPosition( Index row,
                                                             Index column ) const
{

}*/

template< typename Real, typename Index >
bool tnlFastCSRMatrix< Real, tnlHost, Index > :: setElement( Index row,
                                                             Index column,
                                                             const Real& value )
{
	tnlAssert( false, );
	return true;
}

template< typename Real, typename Index >
bool tnlFastCSRMatrix< Real, tnlHost, Index > :: addToElement( Index row,
                                                               Index column,
                                                               const Real& value )
{
	tnlAssert( false, );
	return true;
}

template< typename Real, typename Index >
Index tnlFastCSRMatrix< Real, tnlHost, Index > :: insertColumnSequence( const tnlLongVector< Index >& columns,
	                                                                     Index column_sequence_offset,
		                                                                  Index column_sequence_length,
		                                                                  Index row )
{
}

template< typename Real, typename Index >
bool tnlFastCSRMatrix< Real, tnlHost, Index > :: copyFrom( const tnlCSRMatrix< Real, tnlHost, Index >& csr_matrix )
{
	dbgFunctionName( "tnlFastCSRMatrix< Real, tnlHost >", "copyFrom" );
	if( ! this -> setSize( csr_matrix. getSize() ) )
		return false;

	if( ! this -> setNonzeroElements( csr_matrix. getNonzeroElements() ) )
		return false;

	nonzero_elements = csr_matrix. nonzero_elements;
   column_sequences = csr_matrix. columns;
   row_offsets = csr_matrix. row_offsets;
	last_nonzero_element = csr_matrix. last_nonzero_element;

	const Index compression_depth = 1024;
	for( Index row = 0; row < this -> size; row ++ )
	{
		Index column_sequence_offset = csr_matrix. row_offsets[ row ];
		Index column_sequence_length = csr_matrix. row_offsets[ row + 1 ] - column_sequence_offset;

		/*
		 * Check the sequence from the previous row whether the column sequence is the same
		 */
		bool match( false );

		for( Index cmp_row = row - 1; cmp_row >=0 && cmp_row > row - compression_depth && ! match; cmp_row -- )
		//if( row > 0 )
		{
			Index previous_column_sequence_length = column_sequences_lengths[ cmp_row ];
			if( previous_column_sequence_length == column_sequence_length )
			{
				Index previous_column_sequence_offset = columns_sequences_offsets[ cmp_row ];
				Index j = 0;
				while( j < column_sequence_length &&
					   column_sequences[ previous_column_sequence_offset + j ] == csr_matrix. columns[ column_sequence_offset + j ] - row )
					j ++;
				if( j == column_sequence_length )
				{
					columns_sequences_offsets[ row ] = columns_sequences_offsets[ cmp_row ];
					column_sequences_lengths[ row ] = column_sequence_length;
					match = true;
				}
			}

		}
		if( ! match )
		{
			for( Index i = 0; i < column_sequence_length; i ++ )
				column_sequences[ column_sequence_offset + i ] =
						csr_matrix. columns[ column_sequence_offset + i ] - row;
			columns_sequences_offsets[ row ] = column_sequence_offset;
			column_sequences_lengths[ row ] = column_sequence_length;
			column_sequences_length += column_sequence_length;
		}
	}
	//cout << "Column sequences compression is " << column_sequences_length << "/" << csr_matrix. columns. getSize() << endl;
	return true;
}

template< typename Real, typename Index >
Real tnlFastCSRMatrix< Real, tnlHost, Index > :: getElement( Index row,
                                                      Index column ) const
{
   tnlAssert( 0 <= row && row < this -> getSize(),
			  cerr << "The row is outside the matrix." );
   tnlAssert( 0 <= column && column < this -> getSize(),
			  cerr << "The column is outside the matrix." );
   Index column_offset = columns_sequences_offsets[ row ];
   Index data_offset = row_offsets[ row ];
   Index row_length = row_offsets[ row + 1 ] - data_offset;
   const Index* cols = column_sequences. getVector();

   Index i = 0;
   while( i < row_length && cols[ column_offset + i ] + row < column ) i ++;

   const Real* els = nonzero_elements. getVector();
   if( i < row_length && cols[ column_offset + i ] + row == column )
	  return els[ data_offset + i ];
   return Real( 0.0 );
}

template< typename Real, typename Index >
Real tnlFastCSRMatrix< Real, tnlHost, Index > :: rowProduct( Index row,
                                                             const tnlLongVector< Real, tnlHost, Index >& vec ) const
{
   tnlAssert( 0 <= row && row < this -> getSize(),
			  cerr << "The row is outside the matrix." );
   tnlAssert( vec. getSize() == this -> getSize(),
              cerr << "The matrix and vector for multiplication have different sizes. "
                   << "The matrix size is " << this -> getSize() << "."
                   << "The vector size is " << vec. getSize() << endl; );

   Index data_offset = row_offsets[ row ];
   Index column_offset =  columns_sequences_offsets[ row ];
   Index row_length = row_offsets[ row + 1 ] - data_offset;

   Index i = 0;
   const Index* cols = column_sequences. getVector();
   const Real* els = nonzero_elements. getVector();
   Real product( 0.0 );

   while( i < row_length )
   {
      product += els[ data_offset + i ] * vec[ cols[ column_offset + i ] + row ];
         i ++;
   }


   Index cols_bytes_cnt( 0 );
   Index val_bytes_cnt( 0 );

   return product;
}

template< typename Real, typename Index >
void tnlFastCSRMatrix< Real, tnlHost, Index > :: vectorProduct( const tnlLongVector< Real, tnlHost, Index >& vec,
                                                                tnlLongVector< Real, tnlHost, Index >& result ) const
{
   tnlAssert( vec. getSize() == this -> getSize(),
              cerr << "The matrix and vector for a multiplication have different sizes. "
                   << "The matrix size is " << this -> getSize() << "."
                   << "The vector size is " << vec. getSize() << endl; );
   tnlAssert( result. getSize() == this -> getSize(),
              cerr << "The matrix and result vector of a multiplication have different sizes. "
                   << "The matrix size is " << this -> getSize() << "."
                   << "The vector size is " << result. getSize() << endl; );

   const Index* cols = column_sequences. getVector();
   const Real* els = nonzero_elements. getVector();

   for( Index row = 0; row < this -> size; row ++ )
   {
      Index data_offset = row_offsets[ row ];
      Index column_offset =  columns_sequences_offsets[ row ];
      Index row_length = row_offsets[ row + 1 ] - data_offset;

      Real product( 0.0 );

      for( Index i = 0; i < row_length; i ++ )
        product += els[ data_offset ++ ] * vec[ cols[ column_offset ++ ] + row ];
      /*Index i = 0;
      while( i < row_length )
         product += els[ data_offset + i ] * vec[ cols[ column_offset + i ++ ] + row ];*/

      result[ row ] = product;
   }
};


template< typename Real, typename Index >
Real tnlFastCSRMatrix< Real, tnlHost, Index > :: getRowL1Norm( Index row ) const
{
	tnlAssert( false, );
};

template< typename Real, typename Index >
void tnlFastCSRMatrix< Real, tnlHost, Index > :: multiplyRow( Index row, const Real& value )
{
	tnlAssert( false, );
};



template< typename Real, typename Index >
bool tnlFastCSRMatrix< Real, tnlHost, Index > :: Save( ostream& file ) const
{
	tnlAssert( false, );
	return true;
};


template< typename Real, typename Index >
bool tnlFastCSRMatrix< Real, tnlHost, Index > :: Load( istream& file )
{
	tnlAssert( false, );
	return true;
};

template< typename Real, typename Index >
void tnlFastCSRMatrix< Real, tnlHost, Index > :: printOut( ostream& str,
		                                            const Index lines ) const
{
   str << "Structure of tnlFastCSRMatrix" << endl;
   str << "Matrix name:" << this -> getName() << endl;
   str << "Matrix size:" << this -> getSize() << endl;
   str << "Allocated elements:" << nonzero_elements. getSize() << endl;
   str << "Matrix rows:" << endl;
   Index print_lines = lines;
   if( ! print_lines )
	   print_lines = this -> getSize();

   for( Index i = 0; i < print_lines; i ++ )
   {
      Index first = row_offsets[ i ];
      Index last = row_offsets[ i + 1 ];
      Index column_offset = columns_sequences_offsets[ i ];
      Index column_length = column_sequences_lengths[ i ];
      str << " Row number " << i
          << " , Elements " << first
          << " -- " << last
          << " Col. Seq. Offset: " << column_offset
          << " Col. Seq. Length: " << column_length << endl;
      str << " Data:   ";
      for( Index j = first; j < last; j ++ )
         str << setprecision( 5 ) << setw( 8 ) << nonzero_elements[ j ] << " ";
      str << endl;
      str << "Columns: ";
      for( Index j = 0; j < column_length; j ++ )
         str << setw( 8 ) << column_sequences[ column_offset + j ] + i << " ";
      str << endl;
   }
};



#endif /* TNLFASTCSRMATRIX_H_ */
