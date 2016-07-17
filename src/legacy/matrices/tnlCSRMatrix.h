/***************************************************************************
                          tnlCSRMatrix.h  -  description
                             -------------------
    begin                : Jul 10, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLCSRMATRIX_H_
#define TNLCSRMATRIX_H_

#include <string>

#include <iostream>
#include <iomanip>
#include <core/vectors/tnlVector.h>
#include <core/tnlAssert.h>
#include <core/mfuncs.h>
#include <matrices/tnlMatrix.h>
#include <debug/tnlDebug.h>

using namespace std;

template< typename Real, typename device, typename Index > class tnlRgCSRMatrix;
template< typename Real, typename device, typename Index > class tnlCusparseCSRMatrix;
template< typename Real, typename device, typename Index > class tnlAdaptiveRgCSRMatrix;
template< typename Real, typename device, typename Index > class tnlFastCSRMatrix;
template< typename Real, typename device, typename Index > class tnlEllpackMatrix;

//! Matrix storing the non-zero elements in the CSR (Compressed Sparse Row) format
/*! For details see. Yousef Saad, Iterative Methods for Sparse Linear Systems, p. 85
    at http://www-users.cs.umn.edu/~saad/ .
    The non-zero elements values are stored in the vector non_zero_elements. Their columns
    are stored in the vector columns on the same position. We also need to know the positions
    of the first non-zero element of the row in the non_zero_elements vector. It is stored
    in the vector row_offsets.
    \author Tomas Oberhuber.
 */

// TODO: add CUDA support
template< typename Real, typename Device = tnlHost, typename Index = int >
class tnlCSRMatrix : public tnlMatrix< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   //! Basic constructor
   tnlCSRMatrix( const tnlString& name );

   const tnlString& getMatrixClass() const;

   tnlString getType() const;

   //! Sets the number of row and columns.
   bool setSize( Index new_size );

   bool setLike( const tnlCSRMatrix< Real, Device, Index >& matrix );

   void reset();

   //! Allocate memory for the nonzero elements.
   bool setNonzeroElements( Index elements );

   Index getNonzeroElements() const;

   Index getNonzeroElementsInRow( const Index& row ) const;

   //! This method explicitly computes the number of the non-zero elements.
   Index checkNonzeroElements() const;

   //! Insert one row to the matrix.
   /*! If there is some data already in this @param row it will be rewritten.
    *  @param elements says number of non-zero elements which will be inserted.
    *  @param data is pointer to the elements values.
    *  @param first_column is the column of the first non-zero element.
    *  @param offsets is a pointer to field with offsets of the elements with
    *  respect to the first one. All of them must sorted increasingly.
    *  The elements which do not fit to the matrix are omitted.
    */
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

   Real getElement( Index row,
                    Index column ) const;

   Index getRowLength( const Index row ) const;

   const Index* getRowColumnIndexes( const Index row ) const;

   const Real* getRowValues( const Index row ) const;

   Real rowProduct( Index row,
                    const tnlVector< Real, Device, Index >& vector ) const;

   template< typename Vector1, typename Vector2 >
   void vectorProduct( const Vector1& x,
                       Vector2& b ) const;

   void setBackwardSpMV( bool backwardSpMV );

   bool performSORIteration( const Real& omega,
                             const tnlVector< Real, Device, Index >& b,
                             tnlVector< Real, Device, Index >& x,
                             Index firstRow,
                             Index lastRow ) const;

   Real getRowL1Norm( Index row ) const;

   bool reorderRows( const tnlVector< Index, Device, Index >& rowPermutation,
                     const tnlCSRMatrix< Real, Device, Index >& csrMatrix );

   void multiplyRow( Index row, const Real& value );

   bool read( istream& str, int verbose = 0 );

   //! Method for saving the matrix to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the matrix from a file
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   template< typename Real2 >
   tnlCSRMatrix< Real, Device, Index >& operator = ( const tnlCSRMatrix< Real2, Device, Index >& csrMatrix );

   //! Prints out the matrix structure
   void printOut( ostream& str,
                  const tnlString& format = tnlString( "" ),
		            const Index lines = 0 ) const;

   //! This method tells the minimum, the maximum and the average number of non-zero elements in row.
   void getRowStatistics( Index& min_row_length,
                          Index& max_row_length,
                          Index& average_row_length ) const;



   protected:

   enum elementOperation { set_element, add_to_element };

   //! This method creates free space for inserting new elements.
   /*!
    * It shifts data in nonzero_elements and columns. It also
    * fixes new row_offsets and last_nonzero_elements. The last_non_zero_element
    * is increased by shift and all row_offsets larger or equal to the position as well.
    * It returns false if there is not enough allocated memory for the shift.
    */
   bool shiftElements( Index position,
                       Index row,
                       Index shift );

   //! Returns the position of the element in the vector of the non-zero elements.
   /*!
    * If there is no such element the returned value points to the next element with
    * higher column index.
    */
   Index getElementPosition( Index row,
                             Index colum ) const;


   //! This is auxiliary functions for setting an element or adding a number to an element.
   /*!
    * It works in both cases when the element already exists or not. If not it creates a new one.
    * If the @param operation equals add_to_element the value is added.
    */
   bool setElementAux( Index row,
                       Index column,
                       const Real& value,
                       const elementOperation operation );

   //! Insert element into preallocated row.
   /*
    * The row offsets must be set properly. This method does not check anything.
    * So one must ne sure about what he is doing!
    */
   void insertToAllocatedRow( Index row,
                              Index column,
                              const Real& value,
                              Index insertedElements = 0 );

   void writePostscriptBody( ostream& str,
                             const int elementSize,
                             bool verbose ) const;


   tnlVector< Real, Device, Index > nonzero_elements;

   tnlVector< Index, Device, Index > columns;

   tnlVector< Index, Device, Index > row_offsets;

   //! The last non-zero element is at the position last_non_zero_element - 1
   Index last_nonzero_element;

   /*!***
    * In floating point arithmetics there can be significant difference if we
    * multiply matrix and vector with columns indexed increasingly and decreasingly.
    * One can switch these two approaches by setting the following variable.
    */
   bool backwardSpMV;

   template< typename Real2, typename Device2, typename Index2 >
      friend class tnlCSRMatrix;
   friend class tnlMatrix< Real, tnlHost, Index >;
   friend class tnlMatrix< Real, tnlCuda, Index >;
   friend class tnlCusparseCSRMatrix< Real, tnlCuda, Index >;
   friend class tnlRgCSRMatrix< Real, tnlHost, Index >;
   friend class tnlRgCSRMatrix< Real, tnlCuda, Index >;
   friend class tnlAdaptiveRgCSRMatrix< Real, tnlHost, Index >;
   friend class tnlAdaptiveRgCSRMatrix< Real, tnlCuda, Index >;
   friend class tnlFastCSRMatrix< Real, tnlHost, Index >;
   friend class tnlEllpackMatrix< Real, tnlHost, Index >;
};

template< typename Real, typename Device, typename Index >
tnlCSRMatrix< Real, Device, Index > :: tnlCSRMatrix( const tnlString& name )
   : tnlMatrix< Real, Device, Index >( name ),
     nonzero_elements( name + " : nonzero-elements" ),
     columns( name + " : columns" ),
     row_offsets( name + " : row_offsets" ),
     last_nonzero_element( 0 ),
     backwardSpMV( false )
{
};

template< typename Real, typename Device, typename Index >
const tnlString& tnlCSRMatrix< Real, Device, Index > :: getMatrixClass() const
{
   return tnlMatrixClass :: main;
};

template< typename Real, typename Device, typename Index >
tnlString tnlCSRMatrix< Real, Device, Index > :: getType() const
{
   return tnlString( "tnlCSRMatrix< ") +
           tnlString( ::getType< Real >() ) +
           tnlString( ", " ) +
           Device :: getDeviceType() +
           tnlString( " >" );
};

template< typename Real, typename Device, typename Index >
bool tnlCSRMatrix< Real, Device, Index > :: setSize( Index new_size )
{
   this->size = new_size;
   if( ! row_offsets. setSize( this->size + 1 ) )
      return false;
   row_offsets. setValue( 0 );
   last_nonzero_element = 0;
   return true;
};

template< typename Real, typename Device, typename Index >
bool tnlCSRMatrix< Real, Device, Index > :: setLike( const tnlCSRMatrix< Real, Device, Index >& matrix )
{
   dbgFunctionName( "tnlCSRMatrix< Real, Device, Index >", "setLike" );
   dbgCout( "Setting size to " << matrix. getSize() << "." );

   this->size = matrix. getSize();
   if( ! nonzero_elements. setLike( matrix. nonzero_elements ) ||
       ! columns. setLike( matrix. columns ) ||
       ! row_offsets. setLike( matrix. row_offsets ) )
      return false;
   row_offsets. setValue( 0 );
   last_nonzero_element = 0;
   return true;
}

template< typename Real, typename Device, typename Index >
void tnlCSRMatrix< Real, Device, Index > :: reset()
{
   nonzero_elements. reset();
   columns. reset();
   row_offsets. reset();
   last_nonzero_element = 0;
}

template< typename Real, typename Device, typename Index >
bool tnlCSRMatrix< Real, Device, Index > :: setNonzeroElements( Index elements )
{
   if( ! nonzero_elements. setSize( elements ) )
      return false;
   nonzero_elements. setValue( (Real) 0 );
   if( ! columns. setSize( elements ) )
      return false;
   columns. setValue( -1 );
   return true;
};

template< typename Real, typename Device, typename Index >
Index tnlCSRMatrix< Real, Device, Index > :: getNonzeroElements() const
{
	return nonzero_elements. getSize();
}

template< typename Real, typename Device, typename Index >
Index tnlCSRMatrix< Real, Device, Index > :: getNonzeroElementsInRow( const Index& row ) const
{
   tnlAssert( row >= 0 && row < this->getSize(),
              cerr << "row = " << row << " this->getSize() = " << this->getSize() );
   return row_offsets[ row + 1 ] - row_offsets[ row ];
}

template< typename Real, typename Device, typename Index >
Index tnlCSRMatrix< Real, Device, Index > :: checkNonzeroElements() const
{
	Index elements( 0 );
	for( Index i = 0; i < nonzero_elements. getSize(); i ++ )
		elements += ( nonzero_elements[ i ] != 0 );
	return elements;
}

template< typename Real, typename Device, typename Index >
Index tnlCSRMatrix< Real, Device, Index > :: getRowLength( const Index row ) const
{
	tnlAssert( row >= 0 && row < this->getSize(), );
	return row_offsets[ row + 1 ] - row_offsets[ row ];
}

template< typename Real, typename Device, typename Index >
bool tnlCSRMatrix< Real, Device, Index > :: shiftElements( Index position,
                                                           Index row,
                                                           Index shift )
{
   dbgFunctionName( "tnlCSRMatrix< Real, Device, Index >", "shiftElements" );
   dbgCout( "Shifting non-zero elements by " << shift << " elements." );
   tnlAssert( position <= last_nonzero_element,
              cerr << "Starting position for the shift in tnlCSRMatrix is behind the last element." );
   if( last_nonzero_element + shift > nonzero_elements. getSize() )
   {
      cerr << "Not enough allocated memory to shift the data in tnlCSRMatrix." << endl;
      return false;
   }
   if( position + shift < 0 )
   {
      cerr << "Attempt to shift to negative values." << endl;
      return false;
   }
   if( shift == 0 )
      return true;

   Real* els = nonzero_elements. getData();
   Index* cls = columns. getData();
   if( shift > 0 )
   {
      for( Index i = last_nonzero_element - 1; i >= position; i -- )
      {
         els[ i + shift ] = els[ i ];
         cls[ i + shift ] = cls[ i ];
      }
      for( Index i = position; i < position + shift; i++ )
      {
         els[ i ] = 0.0;
         cls[ i ] = -1;
      }
   }
   else // if( shift > 0 ) - note shift must be < 0 now
   {
      for( Index i = position; i < last_nonzero_element; i ++ )
      {
         els[ i + shift ] = els[ i ];
         cls[ i + shift ] = cls[ i ];
      }
   }
   last_nonzero_element += shift;
   for( Index i = row + 1; i <= this->size; i ++ )
      if( row_offsets[ i ] >= position )
      {
         row_offsets[ i ] += shift;
         if( i > 1 && row_offsets[ i ] < row_offsets[ i -1 ] )
         {
            cerr << "The shift in the tnlCSRMatrix lead to the row pointer crossing for rows "
                 << i - 1 << " and " << i << "." << endl;
            return false;
         }
      }
   return true;
}

template< typename Real, typename Device, typename Index >
bool tnlCSRMatrix< Real, Device, Index > :: insertRow( Index row,
                                                       Index elements,
                                                       Real* data,
                                                       Index first_column,
                                                       Index* offsets )
{
   dbgFunctionName( "tnlCSRMatrix< Real, Device, Index >", "insertRow" )
   tnlAssert( row >=0 && row < this->getSize(),
              cerr << "The row " << row << " is out of the matrix." );
   tnlAssert( elements > 0,
              cerr << "The number of elements to insert is negative:" << elements << "." );
   tnlAssert( data != NULL,
              cerr << "Null pointer passed as data for the inserted row." );
   tnlAssert( first_column >=0 &&  first_column < this->getSize(),
              cerr << "first_column is out of the matrix" );
   tnlAssert( offsets != NULL,
              cerr << "Null pointer passed as data for the column offsets." );

   /*
    *  Cut off elements which do not fit into the matrix.
    *  We first cut off those which have column smaller then zero.
    */
   while( elements > 0 && first_column + offsets[ 0 ] < 0 )
   {
	   elements --;
	   offsets ++;
	   data ++;
	   dbgCout( "Decreasing elements to " << elements << " increasing offsets to element " << offsets[ 0 ] << "." );
   }
   /*
    * And now those which have column larger then size - 1
    */
   while( elements > 0 && first_column + offsets[ elements - 1 ] >= this->size )
   {
	   elements --;
	   dbgCout( "Decreasing elements to " << elements << "." );
   }

   /*
    * Now we shift the data behind this row if it is necessary.
    */
   Index first_in_row = row_offsets[ row ];
   Index current_row_elements = row_offsets[ row + 1 ] - first_in_row;
   if( ! shiftElements( first_in_row, row, elements - current_row_elements ) )
      return false;

   /*
    * And now we insert the data.
    */
   Real* els = nonzero_elements. getData();
   Index* cls = columns. getData();
   for( Index i = 0; i < elements; i ++ )
   {
	  Index column = first_column + offsets[ i ];
     els[ first_in_row + i ] = data[ i ];
     cls[ first_in_row + i ] = column;
     if( i > 0 && offsets[ i - 1 ] >= offsets[ i ] )
     {
        cerr << "The offsets of the elements inserted to the tnlCSRMatrix are not sorted monotonicaly increasingly." << endl;
        return false;
     }
   }
   return true;
}

template< typename Real, typename Device, typename Index >
Index tnlCSRMatrix< Real, Device, Index > :: getElementPosition( Index row,
                                                                  Index column ) const
{
   tnlAssert( 0 <= row && row < this->getSize(),
              cerr << "The row is outside the matrix." );
   tnlAssert( 0 <= column && column < this->getSize(),
              cerr << "The column is outside the matrix." );
   Index first_in_row = row_offsets[ row ];
   Index last_in_row = row_offsets[ row + 1 ];
   const Index* cols = columns. getData();
   Index i = first_in_row;
   while( i < last_in_row && cols[ i ] < column ) i++;
   return i;
}


template< typename Real, typename Device, typename Index >
bool tnlCSRMatrix< Real, Device, Index > :: setElementAux( Index row,
                                                            Index column,
                                                            const Real& value,
                                                            const elementOperation operation )
{
   dbgFunctionName( "tnlCSRMatrix< Real, Device, Index >", "setElementAux" );
   Real* els = nonzero_elements. getData();
   Index* cols = columns. getData();
   Index i = getElementPosition( row, column );
   dbgCout( "Element position is " << i );
   if( cols[ i ] == column && i < row_offsets[ row + 1 ] )
      els[ i ] = value;
   else
   {
      if( ! shiftElements( i, row, 1 ) )
         return false;
      cols[ i ] = column;
      els[ i ] = value;
   }
   return true;
}

template< typename Real, typename Device, typename Index >
bool tnlCSRMatrix< Real, Device, Index > :: setElement( Index row,
                                                         Index column,
                                                         const Real& value )
{
   return setElementAux( row, column, value, set_element );
}

template< typename Real, typename Device, typename Index >
bool tnlCSRMatrix< Real, Device, Index > :: addToElement( Index row,
                                                           Index column,
                                                           const Real& value )
{
   return setElementAux( row, column, value, add_to_element );
}

template< typename Real, typename Device, typename Index >
Real tnlCSRMatrix< Real, Device, Index > :: getElement( Index row,
                                                        Index column ) const
{
   const Real* els = nonzero_elements. getData();
   const Index* cols = columns. getData();
   Index i = getElementPosition( row, column );
   if( i < row_offsets[ row + 1 ] && cols[ i ] == column )
      return els[ i ];
   return Real( 0.0 );
}

template< typename Real, typename Device, typename Index >
const Index* tnlCSRMatrix< Real, Device, Index > :: getRowColumnIndexes( const Index row ) const
{
   return &columns[ row_offsets[ row ] ];
}

template< typename Real, typename Device, typename Index >
const Real* tnlCSRMatrix< Real, Device, Index > :: getRowValues( const Index row ) const
{
   return &nonzero_elements[ row_offsets[ row ] ];
}

template< typename Real, typename Device, typename Index >
Real tnlCSRMatrix< Real, Device, Index > :: rowProduct( Index row,
                                                         const tnlVector< Real, Device, Index >& vec ) const
{
   tnlAssert( 0 <= row && row < this->getSize(),
              cerr << "The row is outside the matrix." );
   tnlAssert( vec. getSize() == this->getSize(),
              cerr << "The matrix and vector for multiplication have different sizes. "
                   << "The matrix size is " << this->getSize() << "."
                   << "The vector size is " << vec. getSize() << endl; );
   Index i = row_offsets[ row ];
   Index last_in_row = row_offsets[ row + 1 ];
   const Index* cols = columns. getData();
   const Real* els = nonzero_elements. getData();
   Real product( 0.0 );
   while( i < last_in_row )
   {
      product += els[ i ] * vec[ cols[ i ] ];
      i ++;
   }
   return product;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector1, typename Vector2 >
void tnlCSRMatrix< Real, Device, Index > :: vectorProduct( const Vector1& vec,
                                                           Vector2& result ) const
{
   tnlAssert( vec. getSize() == this->getSize(),
              cerr << "The matrix and vector for a multiplication have different sizes. "
                   << "The matrix size is " << this->getSize() << "."
                   << "The vector size is " << vec. getSize() << endl; );
   tnlAssert( result. getSize() == this->getSize(),
              cerr << "The matrix and result vector of a multiplication have different sizes. "
                   << "The matrix size is " << this->getSize() << "."
                   << "The vector size is " << result. getSize() << endl; );

   const Index* cols = columns. getData();
   const Index* rw_offsets = row_offsets. getData();
   const Real* els = nonzero_elements. getData();

   if( ! backwardSpMV )
   {
//#ifdef HAVE_OPENMP
//#pragma omp parallel for
//#endif
      for( Index row = 0; row < this->size; row ++ )
      {
         Real product( 0.0 );
         Index i = rw_offsets[ row ];
         Index last_in_row = rw_offsets[ row + 1 ];
         while( i < last_in_row )
         {
            product += els[ i ] * vec. getElement( cols[ i ] );
            i ++;
         }
         result. setElement( row, product );
      }
   }
   else
   {
//#ifdef HAVE_OPENMP
//#pragma omp parallel for
//#endif
      for( Index row = 0; row < this->size; row ++ )
      {
         Real product( 0.0 );
         Index i = rw_offsets[ row + 1 ] - 1;
         Index first_in_row = rw_offsets[ row ];
         while( i >= first_in_row )
         {
            product += els[ i ] * vec. getElement( cols[ i ] );
            i --;
         }
         result. setElement( row, product );
      }
   }
}

template< typename Real, typename Device, typename Index >
bool tnlCSRMatrix< Real, Device, Index > :: performSORIteration( const Real& omega,
                                                                 const tnlVector< Real, Device, Index >& b,
                                                                 tnlVector< Real, Device, Index >& x,
                                                                 Index firstRow,
                                                                 Index lastRow ) const
{
   tnlAssert( firstRow >=0 && firstRow < this->getSize(),
              cerr << "Wrong parameter firstRow. Should be in 0..." << this->getSize()
                   << " but it equals " << firstRow << endl; );
   tnlAssert( lastRow >=0 && lastRow < this->getSize(),
              cerr << "Wrong parameter lastRow. Should be in 0..." << this->getSize()
                   << " but it equals " << lastRow << endl; );

   if( lastRow == 0 )
      lastRow = this->getSize();
   for( Index i = firstRow; i < lastRow; i ++ )
   {
      Real diagonal( 0.0 );
      Real update = b[ i ];
      for( Index j = this->row_offsets[ i ]; j < this->row_offsets[ i + 1 ]; j ++ )
      {
         const Index column = this->columns[ j ];
         if( column == i )
            diagonal = this->nonzero_elements[ j ];
         else
            update -= this->nonzero_elements[ j ] * x[ column ];
      }
      if( diagonal == ( Real ) 0.0 )
      {
         cerr << "There is zero on the diagonal in " << i << "-th row. I cannot perform SOR iteration." << endl;
         return false;
      }
      x[ i ] = ( ( Real ) 1.0 - omega ) * x[ i ] + omega / diagonal * update;
   }
   return true;
}


template< typename Real, typename Device, typename Index >
void tnlCSRMatrix< Real, Device, Index > :: setBackwardSpMV( bool backwardSpMV )
{
   this->backwardSpMV = backwardSpMV;
}

template< typename Real, typename Device, typename Index >
Real tnlCSRMatrix< Real, Device, Index > :: getRowL1Norm( Index row ) const
{
   tnlAssert( 0 <= row && row < this->getSize(),
                 cerr << "The row is outside the matrix." );
   Index i = row_offsets[ row ];
   Index last_in_row = row_offsets[ row + 1 ];
   const Real* els = nonzero_elements. getData();
   Real norm( 0.0 );
   while( i < last_in_row )
      norm += fabs( els[ i ] );
   return norm;
};

template< typename Real, typename Device, typename Index >
void tnlCSRMatrix< Real, Device, Index > :: multiplyRow( Index row, const Real& value )
{
   tnlAssert( 0 <= row && row < this->getSize(),
                 cerr << "The row is outside the matrix." );
   Index i = row_offsets[ row ];
   Index last_in_row = row_offsets[ row + 1 ];
   Real* els = nonzero_elements. getData();
   while( i < last_in_row )
      els[ i ] *= value;
};

template< typename Real, typename Device, typename Index >
bool tnlCSRMatrix< Real, Device, Index > :: reorderRows( const tnlVector< Index, Device, Index >& rowPermutation,
                                                         const tnlCSRMatrix< Real, Device, Index >& inputCsrMatrix )
{
   dbgFunctionName( "tnlCSRMatrix< Real, Device, Index >", "reorderRows" );
   last_nonzero_element = 0;
   if( ! this->setLike( inputCsrMatrix ) )
   {
      cerr << "I am not able to allocate new memory for matrix reordering." << endl;
      return false;
   }
   for( Index i = 0; i < this->getSize(); i ++ )
   {
      tnlAssert( last_nonzero_element < nonzero_elements. getSize(), );
      tnlAssert( last_nonzero_element < columns. getSize(), );
      row_offsets[ i ] = last_nonzero_element;
      Index row = rowPermutation[ i ];
      Index j = inputCsrMatrix. row_offsets[ row ];
      while( j < inputCsrMatrix. row_offsets[ row + 1 ] )
      {
         nonzero_elements[ last_nonzero_element ] = inputCsrMatrix. nonzero_elements[ j ];
         columns[ last_nonzero_element ++ ] = inputCsrMatrix. columns[ j ++ ];
      }
   }
   tnlAssert( last_nonzero_element <= nonzero_elements. getSize(), );
   tnlAssert( last_nonzero_element <= columns. getSize(), );
   row_offsets[ this->getSize() ] = last_nonzero_element;
   dbgExpr( row_offsets[ this->getSize() ] );
   dbgExpr( this->getSize() );
   return true;
}

template< typename Real, typename Device, typename Index >
bool tnlCSRMatrix< Real, Device, Index > :: save( tnlFile& file ) const
{
   if( ! tnlMatrix< Real, Device, Index > :: save( file ) ) return false;
   if( ! nonzero_elements. save( file ) ) return false;
   if( ! columns. save( file ) ) return false;
   if( ! row_offsets. save( file ) ) return false;
#ifdef HAVE_NOT_CXX11
   if( ! file. write< const Index, tnlHost >( &last_nonzero_element ) )
#else
   if( ! file. write( &last_nonzero_element ) )
#endif
      return false;
   return true;
};

template< typename Real, typename Device, typename Index >
bool tnlCSRMatrix< Real, Device, Index > :: load( tnlFile& file )
{
   if( ! tnlMatrix< Real, Device, Index > :: load( file ) ) return false;
   if( ! nonzero_elements. load( file ) ) return false;
   if( ! columns. load( file ) ) return false;
   if( ! row_offsets. load( file ) ) return false;
#ifdef HAVE_NOT_CXX11
   if( ! file. read< Index, tnlHost >( &last_nonzero_element ) )
#else
   if( ! file. read( &last_nonzero_element ) )
#endif
      return false;
   return true;
};

template< typename Real, typename Device, typename Index >
bool tnlCSRMatrix< Real, Device, Index > :: save( const tnlString& fileName ) const
{
   return tnlObject :: save( fileName );
}

template< typename Real, typename Device, typename Index >
bool tnlCSRMatrix< Real, Device, Index > :: load( const tnlString& fileName )
{
   return tnlObject :: load( fileName );
}

template< typename Real, typename Device, typename Index >
   template< typename Real2 >
tnlCSRMatrix< Real, Device, Index >& tnlCSRMatrix< Real, Device, Index > :: operator = ( const tnlCSRMatrix< Real2, Device, Index >& csrMatrix )
{
   if( ! nonzero_elements. setSize( csrMatrix. nonzero_elements. getSize() ) ||
       ! columns. setSize( csrMatrix. columns. getSize() ) ||
       ! row_offsets. setSize( csrMatrix. row_offsets. getSize() ) )
   {
      cerr << "I am unable to allocate memory for new matrix!" << endl;
      abort();
   }
   nonzero_elements = csrMatrix. nonzero_elements;
   columns = csrMatrix. columns;
   row_offsets = csrMatrix. row_offsets;
   tnlMatrix< Real, Device, Index > :: operator = ( csrMatrix );
   return * this;
}

template< typename Real, typename Device, typename Index >
void tnlCSRMatrix< Real, Device, Index > :: printOut( ostream& str,
                                                      const tnlString& name,
                                                      const tnlString& format,
		                                                const Index lines ) const
{
   str << "Structure of tnlCSRMatrix" << endl;
   str << "Matrix name:" << name << endl;
   str << "Matrix size:" << this->getSize() << endl;
   str << "Allocated elements:" << nonzero_elements. getSize() << endl;
   str << "Matrix rows:" << endl;
   Index print_lines = lines;
   if( ! print_lines )
	   print_lines = this->getSize();
   for( Index i = 0; i < print_lines; i ++ )
   {
      Index first = row_offsets[ i ];
      Index last = row_offsets[ i + 1 ];
      str << " Row number " << i
          << " - elements " << first
          << " -- " << last << endl;
      str << " Data:   ";
      for( Index j = first; j < last; j ++ )
         str << setprecision( 5 ) << setw( 8 ) << nonzero_elements[ j ] << " ";
      str << endl;
      str << "Columns: ";
      for( Index j = first; j < last; j ++ )
         str << setw( 8 ) << columns[ j ] << " ";
      str << endl;
      str << "Last non-zero element: " << last_nonzero_element << endl;
   }
};

template< typename Real, typename Device, typename Index >
void tnlCSRMatrix< Real, Device, Index > :: getRowStatistics( Index& min_row_length,
                                                        Index& max_row_length,
                                                        Index& average_row_length ) const
{
   min_row_length = this->getSize();
   max_row_length = 0;
   average_row_length = 0;
   for( Index i = 0; i < this->getSize(); i ++ )
   {
      Index row_length = row_offsets[ i + 1 ] - row_offsets[ i ];
      min_row_length = Min( min_row_length, row_length );
      max_row_length = Max( max_row_length, row_length );
      average_row_length += row_length;
   }
   average_row_length /= ( double ) this->getSize();
};

template< typename Real, typename Device, typename Index >
void tnlCSRMatrix< Real, Device, Index > :: insertToAllocatedRow( Index row,
		                                                    Index column,
		                                                    const Real& value,
		                                                    Index insertedElements )
{
    dbgFunctionName( "tnlCSRMatrix< T >", "insertToAllocatedRow" );

    if( insertedElements != 0 )
    {
       Index element_pos = row_offsets[ row ] + insertedElements - 1;
       tnlAssert( element_pos < row_offsets[ row + 1 ],
                  cerr << "element_pos = " << element_pos
                       << "row_offsets[ row + 1 ] = " << row_offsets[ row + 1 ]
                       << "row_offsets[ row ] = " << row_offsets[ row ]);
       while( element_pos > row_offsets[ row ] &&
              columns[ element_pos ] > column )
       {
          tnlAssert( element_pos + 1 < row_offsets[ row + 1 ], );
          columns[ element_pos + 1 ] = columns[ element_pos ];
          nonzero_elements[ element_pos + 1 ] = nonzero_elements[ element_pos ];
          element_pos --;
          cerr << "*";
       }
       tnlAssert( element_pos >= row_offsets[ row ], );
       //dbgExpr( element_pos );
       if( element_pos == row_offsets[ row ] + insertedElements - 1 )
          element_pos ++;
       columns[ element_pos ] = column;
       nonzero_elements[ element_pos ] = value;
       return;
    }

    Index element_pos = row_offsets[ row ];
    while( element_pos < row_offsets[ row + 1 ] &&
           columns[ element_pos ] < column &&
           columns[ element_pos ] != -1 )
    {
       element_pos ++;
       cerr << "X";
    }
    //dbgExpr( element_pos );

    Index J1( column ), J2;
    double A1( value ), A2;
    if( columns[ element_pos ] == -1 )
    {
       //dbgCout( "Inserting directly...");
       columns[ element_pos ] = J1;
       nonzero_elements[ element_pos ] = A1;
    }
    else
    {
       dbgCout( "Shifting data ..." );
       while( element_pos < row_offsets[ row + 1 ] &&
              columns[ element_pos ] != -1 )
       {
          J2 = columns[ element_pos ];
          A2 = nonzero_elements[ element_pos ];
          columns[ element_pos ] = J1;
          nonzero_elements[ element_pos ] = A1;
          J1 = J2;
          A1 = A2;
          element_pos ++;
       }
    }
}

template< typename Real, typename Device, typename Index >
bool tnlCSRMatrix< Real, Device, Index > :: read( istream& file,
                                                   int verbose )
{
   dbgFunctionName( "tnlCSRMatrix< T >", "read" );
   tnlString line;
   bool dimensions_line( false ), format_ok( false );
   tnlList< tnlString > parsed_line;
   Index non_zero_elements( 0 );
   Index parsed_elements( 0 );
   Index size( 0 );
   tnlVector< Index > rows_length( "rows-length" );
   Index header_end( 0 );
   bool symmetric = false;

   /*
    * For the CSR matrix we read the matrix in two steps.
    * First we compute number of nonzero elements in each row then
    * we allocate necessary memory a set to row pointers. After that
    * we read the file again and store the data.
    */

   /*
    * So first compute the row lengths.
    */
   dbgCout( "Computing the rows length." );
   while( line. getLine( file ) )
   {
      if( ! format_ok )
      {
         format_ok = tnlMatrix< Real, Device, Index > :: checkMtxHeader( line, symmetric );
         continue;
      }
      if( line[ 0 ] == '%' ) continue;
      if( ! format_ok )
      {
         cerr << "Unknown format of the file: " << line << endl;
         cerr << "We expect header line like this:" << endl;
         cerr << "%%MatrixMarket matrix coordinate real general/symmetric" << endl;
         return false;
      }

      if( ! dimensions_line )
      {
         parsed_line. EraseAll();
         line. parse( parsed_line );
         if( parsed_line. getSize() != 3 )
         {
           cerr << "Wrong number of parameters in the matrix header." << endl;
           return false;
         }
         Index M = atoi( parsed_line[ 0 ]. getString() );
         Index N = atoi( parsed_line[ 1 ]. getString() );
         Index L = atoi( parsed_line[ 2 ]. getString() );
         if( symmetric )
        	 L = 2 * L - M;
         if( verbose )
         {
         	cout << "Matrix size:                " << setw( 9 ) << right << M << endl;
         	cout << "Non-zero elements expected: " << setw( 9 ) << right << L << endl;
         }

         if( M <= 0 || N <= 0 || L <= 0 )
         {
           cerr << "Wrong parameters in the matrix header." << endl;
           return false;
         }
         if( M  != N )
         {
           cerr << "There is not square matrix in the file." << endl;
           return false;
         }

         dimensions_line = true;
         non_zero_elements = L;
         size = M;
         header_end = file. tellg();
         rows_length. setSize( size );
         rows_length. setValue( 0 );
         continue;
      }
      if( parsed_line. getSize() != 3 )
      {
         cerr << "Wrong number of parameters in the matrix row at line:" << line << endl;
         return false;
      }

      parsed_line. EraseAll();
      line. parse( parsed_line );
      Index I = atoi( parsed_line[ 0 ]. getString() ) - 1;
      Index J = atoi( parsed_line[ 1 ]. getString() ) - 1;
      if( I < 0 || I >= size )
      {
         cerr << "The row index " << I << " is out of the matrix." << endl;
         return false;
      }
      rows_length[ I ] ++;
      parsed_elements ++;
      if( symmetric && I != J )
      {
    	  rows_length[ J ] ++;
    	  parsed_elements ++;
      }
   }

   if( verbose )
      cout << "Non-zero elements parsed:   " << setw( 9 ) << right << parsed_elements << endl;

   if( ! this->setSize( size ) ||
       ! this->setNonzeroElements( parsed_elements ) )
   {
      cerr << "Not enough memory to allocate the sparse or the full matrix for testing." << endl;
      return false;
   }
   parsed_elements = 0;


   /*
    * Now set the row length.
    */
   dbgCout( "Setting rows length..." );
   Index row_pointer( 0 );
   for( Index i = 0; i < size; i ++ )
   {
      row_offsets[ i ] = row_pointer;
      row_pointer += rows_length[ i ];
   }
   row_offsets[ size ] = row_pointer;
   last_nonzero_element = row_offsets[ size ];


   /*
    * Now read the file again and insert the non-zero elements.
    */
   dbgCout( "Reading the matrix ..." );
   dbgExpr( header_end );
   file. clear();
   file. seekg( header_end, ios :: beg );
   tnlVector< Index, tnlHost > insertedElementsInRows( "tnlCSRMatrix::insertedElementsInRows" );
   insertedElementsInRows. setSize( size );
   insertedElementsInRows. setValue( 0 );
   while( line. getLine( file ) )
   {
      parsed_line. EraseAll();
      line. parse( parsed_line );
      if( parsed_line. getSize() != 3 )
      {
         cerr << "Wrong number of parameters in the matrix row at line:" << line << endl;
         return false;
      }
      Index I = atoi( parsed_line[ 0 ]. getString() ) - 1;
      Index J = atoi( parsed_line[ 1 ]. getString() ) - 1;
      Real A = ( Real ) atof( parsed_line[ 2 ]. getString() );

      if( I < 0 || I >= size || J < 0 || J >= size )
      {
         cerr << "Index outside of the matrix at line: " << line << endl;
         return false;
      }

      //dbgCout( "Inserting element A(" << I << "," << J << ")=" << A );
      insertToAllocatedRow( I, J, A, insertedElementsInRows[ I ] );
      insertedElementsInRows[ I ] ++;
      parsed_elements ++;
      if( symmetric && I != J )
      {
         //dbgCout( "Inserting symmetric element A(" << J << "," << I << ")=" << A );
    	   insertToAllocatedRow( J, I, A, insertedElementsInRows[ J ] );
    	   insertedElementsInRows[ J ] ++;
    	   parsed_elements ++;
      }

      if( verbose )
    	 cout << "Parsed elements:            " << setw( 9 ) << right << parsed_elements << "\r" << flush;

   }
   return true;
}

template< typename Real, typename Device, typename Index >
void tnlCSRMatrix< Real, Device, Index > :: writePostscriptBody( ostream& str,
                                                                 const int elementSize,
                                                                 bool verbose ) const
{
   const double scale = elementSize * this->getSize();
   double hx = scale / ( double ) this->getSize();
   Index lastRow( 0 ), lastColumn( 0 );
   for( Index row = 0; row < this->getSize(); row ++ )
   {
      for( Index i = this->row_offsets[ row ]; i < this->row_offsets[ row + 1 ]; i ++ )
      {
         Real elementValue = this->nonzero_elements[ i ];
         if(  elementValue != ( Real ) 0.0 )
         {
            Index column = this->columns[ i ];
            str << ( column - lastColumn ) * elementSize
                << " " << -( row - lastRow ) * elementSize
                << " translate newpath 0 0 " << elementSize << " " << elementSize << " rectstroke" << endl;
            lastColumn = column;
            lastRow = row;
         }
      }
      if( verbose )
         cout << "Drawing the row " << row << "      \r" << flush;
   }
}


#endif /* TNLCSRMATRIX_H_ */

