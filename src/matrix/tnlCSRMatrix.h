/***************************************************************************
                          tnlCSRMatrix.h  -  description
                             -------------------
    begin                : 2007/07/23
    copyright            : (C) 2007 by Tomá¹ Oberhuber
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

#ifndef tnlCSRMatrixH
#define tnlCSRMatrixH

#include <ostream>
#include <iomanip>
#include <assert.h>
#include <core/mfuncs.h>
#include <matrix/tnlBaseMatrix.h>
#include <debug/tnlDebug.h>

//! Structure for keeping single element of the CSR matrix
/*! This structure stores the element value and its column index.
 */
template< typename T > struct tnlCSRMatrixElement
{
   //! Element value
   T value;
   
   //! Element column
   long int column;

   //! Constructor
   tnlCSRMatrixElement( const T& v, long int col )
   : value( v ), column( col ){};
};

//! Structure for describing single row of the CSR matrix
/*! This structure stores the index of the first and the
    last elements in in the row. 'diagonal' points
    to the diagonal element.
 */
struct tnlCSRMatrixRowInfo
{
   //! Row begining
   long int first;

   //! Last non-zero element of the row
   long int last;

   //! Diagonal element
   /*! It is set to -1 if there is no diagonal element at the row
    */
   long int diagonal;
};

//! Matrix storing the non-zero elements in the CSR (Compressed Sparse Row) format 
/*! For details see. Yousef Saad, Iterative Methods for Sparse Linear Systems, p. 85
    at http://www-users.cs.umn.edu/~saad/ .
    The elements are stored in the array data of the type tnlCSRMatrixElement. It is
    equivalent of the AA and JA arrays in the book. The boundaries of the rows
    (IA array in the book) are stored in the array rows_info.
    Since the number of non-zero elements may increase during the computation, one
    may allocate more memory for elements (it means larger array data). The
    number of allocated elements is stored in allocated_elements. Therefore
    we need to know what is the last relevant element (its index in data array)
    which is stored in last_non_zero_element. One may also reallocate the data array
    during the computation. It is useful in the case when we do not have a good estimate for
    the non-zero elements number at the begining. The number of newly allocated elements is
    stored in param allocation_segment_size.
    \author Tomas Oberhuber.
 */
template< typename T > class tnlCSRMatrix : public tnlBaseMatrix< T >
{
   enum csr_operation { set, add }; 

   public:

   //! Basic constructor
   tnlCSRMatrix()
   : size( 0 ),
     data( 0 ),
     allocated_elements( 0 ),
     allocation_segment_size( 0 ),
     last_non_zero_element( 0 ),
     rows_info( 0 )
#ifdef CSR_MATRIX_TUNING
     ,data_shifts( 0 ),
     data_seeks( 0 ),
     re_allocs( 0 )
#endif
   {
      abort();
   };

   //! The main constructor
   /*! \param _size                     Matrix dimension
       \param _initial_allocation_size  Initial guess for non-zero elements number
       \param _alloctaion_segment_size  If we need to allocate more new non-zero elements this says the amount if increase.
       \param _inititial_row_size       It says how many elements we expect to be in each row. This s information can significialy speedup the insertion of the elements.
    */
   tnlCSRMatrix( const long int _size,
               const long int _initial_allocation = 0,
               const long int _allocation_segment_size = 0,
               const long int _initial_row_size = 0 )
   : size( _size ),
     allocation_segment_size( _allocation_segment_size )
   {
      dbgFunctionName( "tnlCSRMatrix", "tnlCSRMatrix" );
      
      data = ( tnlCSRMatrixElement< T >* ) calloc( _initial_allocation + 1, sizeof( tnlCSRMatrixElement< T >) );
      rows_info = ( tnlCSRMatrixRowInfo* ) calloc( size + 2, sizeof( tnlCSRMatrixRowInfo ) );
      if( ! data || ! rows_info )
      {
         cerr << "Unable to allocate new matrix: " << __FILE__ << " at line " << __LINE__ << "." << endl;
         abort();
      }

      data ++;                  // protection against freeing this memory outside the class
      rows_info ++;

      allocated_elements = _initial_allocation;
      
      if( _initial_row_size < 0 )
      {
         cerr << "Initial row size can not be negative: " << __FILE__ << " at line " << __LINE__ << "." << endl;
         abort();
      }
      
      dbgCout( "Setting rows size to " << _initial_row_size << "( and pointers to diagonal to -1 )" );
      assert( size * _initial_row_size <= allocated_elements );
      long int i;
      for( i = 0; i <= size; i ++ )
      {
         dbgCout( "Setting row " << i << " first and last to " << i * _initial_row_size );
         rows_info[ i ]. first = rows_info[ i ]. last = i * _initial_row_size;
         rows_info[ i ]. diagonal = -1;
      }
      assert( rows_info[ size ]. last <= allocated_elements );

      if( ! allocation_segment_size )
         cerr << "WARNING: Segment size for allocating more memory is set to 0." << endl;

      last_non_zero_element = 0;
   };

   tnlString GetType() const
   {
      T t;
      return tnlString( "tnlCSRMatrix< " ) + tnlString( GetParameterType( t ) ) + tnlString( " >" );
   };

   const tnlString& GetMatrixClass() const
   {
      return tnlMatrixClass :: main;
   };

   //! Direct data acces for constant instances
   /*! This is to make some matrix solver faster. 
       \param _data     Returns the data array (AA and JA arrays).
       \param _rows_info Returns the row_info array (IA array).
   */ 
   void Data( const tnlCSRMatrixElement< T >*& _data, 
              const tnlCSRMatrixRowInfo*& _rows_info ) const
   {
      _data = data;
      _rows_info = rows_info;
   };

   //! Direct data acces
   /*! This is to make some matrix solver faster. 
       \param _data     returns the data array (AA and JA arrays).
       \param _rows_info returns the row_info array (IA array).
   */ 
   void Data( tnlCSRMatrixElement< T >*& _data, 
              const tnlCSRMatrixRowInfo*& _rows_info )
   {
      _data = data;
      _rows_info = rows_info;
   };
   
   //! Size getter
   /*! \return the dimension of the matrix
    */
   long int GetSize() const
   {
      return size;
   };
   
   //! Get element at given row and column.
   /** \param row element row.
    *  \param column element column.
    *  \return value of the element.
    */
   T GetElement( long int row, long int column ) const
   {
      dbgFunctionName( "tnlCSRMatrix", "operator()" );
      dbgCout( "row = " << row << " col = " << column );

      assert( row < size );
      long int row_beg = -1;
      if( column >= row ) row_beg = rows_info[ row ]. diagonal; //diagonal might be also -1
      if( row_beg == -1 ) row_beg = rows_info[ row ]. first;
      long int row_end = rows_info[ row ]. last;
      assert( row_end <= allocated_elements );
      
      dbgCout( "row_beg = " << row_beg << " row_end = " << row_end );

      long int i = row_beg;
      while( i < row_end && data[ i ]. column < column ) i ++;
#ifdef CSR_MATRIX_TUNING
      const_cast< tnlCSRMatrix* >( this ) -> data_seeks += i - row_beg;
#endif
      dbgCout( " i = " << i << " i-th column = " << data[ i ]. column << " value = " << data[ i ]. value );
      if( i < row_end && data[ i ]. column == column ) 
         return data[ i ]. value;
      return 0.0;
   };

   //! Set element at given position
   /*! \return false if some allocation failed.
    */
   bool SetElement( const long int row,
                    const long int col,
                    const T& v )
   {
      dbgFunctionName( "tnlCSRMatrix", "SetElement" );
      dbgCout( "row = " << row << " col = " << col << " value = " << v );
   
      return ChangeElement( row, col, v, set );  
   };
   
   bool AddToElement( long int row, long int column, const T& v )
   {
      if( v == 0.0 ) return true;
      return ChangeElement( row, column, v, add );
   };
   
   //! Set complete row stored as a an array of size equal to the matrix size
   /** THIS METHOD WAS NOT PROPERLY TESTED YET !!!
    *  @param non_zero_elems says how many non-zero elements are there in the array
    */
   bool SetRow( const long int row,
                const T* row_data,
                const long int non_zero_elems,
                const long int first_non_zero,
                const long int last_non_zero )
   {
      dbgFunctionName( "tnlCSRMatrix", "SetSparseRow" );

      long int row_beg = rows_info[ row ]. first;
      long int row_end = rows_info[ row ]. last;
      const long int current_row_size = row_beg - row_end;

      dbgExpr( row_beg );
      dbgExpr( row_end );
      dbgExpr( current_row_size );

      if( current_row_size < non_zero_elems )
      {
         dbgCout( "Shifting the rest of the data" );
         long int shift = non_zero_elems - current_row_size;
         dbgExpr( shift );
         dbgExpr( allocated_elements );
         dbgExpr( rows_info[ size ]. last );
         dbgExpr( size );
         long int new_alloc = shift - ( allocated_elements - rows_info[ size ]. last );
         if( new_alloc > 0 && ! AllocateNewMemory( new_alloc ) ) return false;
         long int j = last_non_zero_element - 1;
         while( j >= row_end ) data[ j + shift ] = data[ j -- ];
         last_non_zero_element += shift;
         j = row + 1;
         while( j <= size )
         {
            rows_info[ j ]. first += shift;
            if( rows_info[ j ]. diagonal != -1 ) rows_info[ j ]. diagonal += shift;
            rows_info[ j ++ ]. last += shift;
         }
      }
      dbgCout( "Setting row end to " << row_beg + non_zero_elems );
      rows_info[ row ]. last = row_beg + non_zero_elems;
      dbgCout( "Reseting diagonal entry pionter." );
      rows_info[ row ]. diagonal = -1;
      long int i( 0 ), j( first_non_zero ), row_pos( row_beg );
      while( i < non_zero_elems && j <= last_non_zero )
      {
         if( row_data[ j ] == 0 )
         {
            j ++;
            continue;
         }
         data[ row_pos ]. value = row_data[ j ];
         data[ row_pos ]. column = j;
         //dbgCout( "Setting row " << row << " col " << row_pos << " to " << row_data[ j ] );
         if( j == row )
         {
            rows_info[ row ]. diagonal = row_pos;
            //dbgCout( "Seting diagonal entry pointer to " << row_pos );
         }
         i ++;
         row_pos ++;
         j ++;
      }
   };

   //! Reset matrix
   /*! \param new_size New matrix dimension.
       \param new_data_size New data array size.
       \param new_segment_size New segment size.
       \return False if some allocation failed.
    */
   bool Reset( long int new_size = 0,
               long int new_initial_allocation = 0,
               long int new_allocation_segment_size = 0,
               long int new_initial_row_size = 0 )
   {
      dbgFunctionName( "tnlCSRMatrix", "Reset" );
      if( new_size && size != new_size )
      {
         rows_info = ( tnlCSRMatrixRowInfo* ) realloc( --rows_info, ( new_size + 1 ) * sizeof( tnlCSRMatrixRowInfo ) );
         if( ! rows_info )
         {
            cerr << "Unable to reallocate new matrix: " << __FILE__ << " at line " << __LINE__ << "." << endl;
            abort();
         }
         rows_info ++;
         size = new_size;
      }

      if( new_initial_allocation && allocated_elements != new_initial_allocation )
      {
         data = ( tnlCSRMatrixElement< T >* ) realloc( --data, ( new_initial_allocation + 1 ) * sizeof( tnlCSRMatrixElement< T > ) );
         if( ! data )
         {
            cerr << "Unable to reallocate new matrix: " << __FILE__ << " at line " << __LINE__ << "." << endl;
            abort();
         }
         data ++;
         allocated_elements = new_initial_allocation;
      }

      long int i;
      if( new_initial_row_size )
      {
         dbgCout( "Setting rows size to " << new_initial_row_size );
         assert( size * new_initial_row_size <= allocated_elements );
         for( i = 0; i <= size; i ++ )
         {
            dbgCout( "Setting row " << i << " beginning to " << i * new_initial_row_size );
            rows_info[ i ]. first = rows_info[ i ]. last = i * new_initial_row_size;
            rows_info[ i ]. diagonal = -1;
         }
      }
      else
      {
         for( i = 0; i <= size; i ++ )
         {
            dbgCout( "Setting row " << i << " beginning to " << i * new_initial_row_size );
            rows_info[ i ]. last = rows_info[ i ]. first;
            rows_info[ i ]. diagonal = -1;
         }
      }

      if( new_allocation_segment_size )
         allocation_segment_size = new_allocation_segment_size;
      last_non_zero_element = 0;
      return true;
   };  

   //! Clone matrix
   bool Clone( const tnlCSRMatrix& m )
   {
      size = m. GetSize();
      rows_info = ( tnlCSRMatrixRowInfo* ) realloc( --rows_info, ( size + 1 ) * sizeof( tnlCSRMatrixRowInfo ) ); 
      if( ! rows_info ) return false;
      rows_info ++;
      
      if( allocated_elements < m. allocated_elements )
      {
         allocated_elements = m. allocated_elements;
         data = ( tnlCSRMatrixElement< T >* ) realloc( --data, allocated_elements * sizeof( tnlCSRMatrixElement< T > ) );
         if( ! data ) return false;
         data ++;
      }
      memcpy( rows_info, m. rows_info, ( size + 1 ) * sizeof( tnlCSRMatrixRowInfo ) );
      last_non_zero_element = m. last_non_zero_element;
      allocation_segment_size = m. allocation_segment_size;
      long int l = Min( last_non_zero_element + 1, allocated_elements );
      long int i;
      for( i = 0; i < l; i ++ ) data[ i ] = m. data[ i ];
      return true;
   };

   //! Row product
   /*! Compute product of given vector with given row
    */
   T RowProduct( const long int row, const T* vec ) const
   {
      dbgFunctionName( "tnlCSRMatrix", "RowProduct" );
      long int row_beg = rows_info[ row ]. first;
      long int row_end = rows_info[ row ]. last;
      dbgCout( "row_beg = " << row_beg << " row_end = " << row_end );
      
      long int col;
      T res( 0.0 );
      long int i = row_beg;
      while( i < row_end )
      {
         assert( data[ i ]. column >= 0 && data[ i ]. column < size );
         res += data[ i ]. value * vec[ data[ i ++ ]. column ];
      }
      return res;
   };

   //! Vector product
   void VectorProduct( const T* vec, T* result ) const
   {
      dbgFunctionName( "tnlCSRMatrix", "VectorProduct" );
      long int row, i;
      T res;
#ifdef HAVE_OPENMP
#pragma omp parallel for private( row, i )
#endif
      for( row = 0; row < size; row ++ )
      {
         const long int row_beg = rows_info[ row ]. first;
         const long int row_end = rows_info[ row ]. last;
         dbgCout( "row = " << row << " row_beg = " << row_beg << " row_end = " << row_end );
         
         res = 0.0;
         i = row_beg;
         while( i < row_end )
            res += data[ i ]. value * vec[ data[ i ++ ]. column ];
         
         result[ row ] = res;
      }
      dbgCout( "VectorProduct done." );
   };
   
   //! Add matrix
   void MatrixAdd( const tnlCSRMatrix& m2 );

   //! Multiply row
   void MultiplyRow( const long int row, const T& c )
   {
      const long int row_beg = rows_info[ row ]. first;
      const long int row_end = rows_info[ row ]. last;
      long int i;
      for( i = row_beg; i < row_end; i ++ )
         data[ i ]. value *= c;
   };

   //! Get row L1 norm
   T GetRowL1Norm( const long int row ) const
   {
      const long int row_beg = rows_info[ row ]. first;
      const long int row_end = rows_info[ row ]. last;
      T res( 0.0 );
      long int i;
      for( i = row_beg; i < row_end; i ++ )
         res += fabs( data[ i ]. value );
      return res;
   };

   //! Print matrix
   /*! This is for the debuging purpose. For nicer output one should use the operator <<.
    */
   void Print( ostream& str )
   {
      str << "Data array:" << endl;
      long int i = 0;
      while( i < last_non_zero_element )
         str << i << "  Col: " << data[ i ]. column << " Val: " << data[ i ++ ]. value << endl;
      str << "Rows info:" << endl;
      i = 0;
      while( i <= size )
         str << " Row " << i << " First " << rows_info[ i ]. first <<
                " Diagonal " << rows_info[ i ]. diagonal <<
                " Last " << rows_info[ i ++ ]. last << endl;
      str << " Last non-zero element " << last_non_zero_element <<
             " Allocated elements " << allocated_elements << endl;
   };

#ifdef CSR_MATRIX_TUNING
   void PrintStatistics()
   {
      cout << "Data seeks: " << data_seeks << endl 
           << "Data shifts: " << data_shifts << endl
           << "Reallocations: " << re_allocs << endl
           << "Allocated elements; " << allocated_elements << endl;
   }
   
   void ResetStatistics()
   {
      data_seeks = data_shifts = re_allocs = 0;
   }
#endif

   //! Destructor
   ~tnlCSRMatrix()
   {
      if( data ) delete[] --data;
      if( rows_info ) delete[] -- rows_info;
   };

   protected:

   //! Allocates new memory for the data array
   /*! If 'allocation_segment_size' is set, the number
       of the newly allocated elements is equal to the smallest multiple of
       'allocation_segment_size' larger then 'amount' - otherwise the number
       of new allocated elements is equal to the 'amount'.
       \param amount positive number of new elements that are requested.
       \return false if the allocation failed otherwise true.
   */
   bool AllocateNewMemory( long int amount )
   {
      assert( amount > 0 );
      dbgFunctionName( "tnlCSRMatrix", "AllocateNewMemory" );
#ifdef CSR_MATRIX_TUNING
      re_allocs ++;
#endif
      dbgCout( "Data array is full! Need to reallocate a new one." );
      dbgCout( "Old data array pointer is " << data );
      if( ! allocation_segment_size )
         allocated_elements += amount;
      else
         allocated_elements += 
            ( ( amount / allocation_segment_size ) + 1 ) * allocation_segment_size;
      data = ( tnlCSRMatrixElement< T >* ) realloc( --data, ( allocated_elements + 1 ) * sizeof( tnlCSRMatrixElement< T > ) );
      if( ! data ) return false;
      data ++;
      return true;
   };
   
   //! Change (set or add to) element at given position.
   /**
    * \param v value to be set or add to given element.
    * \param row element row.
    * \param col element column
    * \param operation can be set or add.
    * \return false if some allocation failed.
    */
   bool ChangeElement( const long int row,
                       const long int col,
                       const T& v,
                       csr_operation operation )
   {
      dbgFunctionName( "tnlCSRMatrix", "ChangeElement" );
      dbgCout( "row = " << row << " col = " << col << " value = " << v );
      //cout <<  "row = " << row << " col = " << col << " value = " << v << endl;
     
      // check whether the input parameters are correct
      if( row < 0 || col < 0 )
      {
         cerr << "Negative row( " << row << " ) or column ( " << col << 
                 " ) in tnlCSRMatrix :: Set calling." << endl;
         abort();
      }
      if( row >= size )
      {
         cerr << "Parametr row exceeds number of matrix rows in tnlCSRMatrix :: Set call." << endl;
         abort();
      }

      dbgCout( "Check if given element is already set" );
      long int _row_beg = -1;
      if( col >= row ) _row_beg = rows_info[ row ]. diagonal; //diagonal might be also -1
      if( _row_beg == -1 ) _row_beg = rows_info[ row ]. first;
      const long int row_beg = _row_beg;
      const long int row_end = rows_info[ row ]. last;
      assert( row_end <= allocated_elements );
      long int i = row_beg;
      while( i < row_end && data[ i ]. column < col ) i ++;
#ifdef CSR_MATRIX_TUNING
         data_seeks += i - row_beg;
#endif

      if( i < row_end && data[ i ]. column == col ) // given element was found 
      {
         dbgCout( "Element " << row << ", " << col << " was found." );
         assert( i < allocated_elements );
         if( operation == set )
            data[ i ]. value = v;
         else data[ i ]. value += v;
         return true; 
      }
      assert( GetElement( row, col ) == 0.0 );
      // given element does not exist 
      // if the value to be set is zero we will do nothing
      if( v == 0.0 ) return true;

      // otherwise we must set the new element
      dbgCout( "Element was not set yet" );
      //cout <<  "Element was not set yet" << endl;
      const long int next_row_beg = rows_info[ row + 1 ]. first;
      if( row_end < next_row_beg ) // we can insert the element in this row without shifting the rest
      {
         dbgCout( "Free space in the row " << row << " found - inserting new element." );
         //cout <<  "Free space in the row " << row << " found - inserting new element." << endl;
         dbgExpr( row_end );
         long int j = row_end;
#ifdef CSR_MATRIX_TUNING
         data_shifts += j - i;
#endif
         while( j > i )
         {
            data[ j ] = data[ j - 1 ];
            j --;
         }
         data[ i ]. column = col;
         data[ i ]. value = v;
         if( col < row && rows_info[ row ]. diagonal != -1 ) rows_info[ row ]. diagonal ++;
         if( row == col )
         {
            dbgCout( "Setting pointer to diagonal element on row " << row << " to " << i );
            rows_info[ row ]. diagonal = i;
         }
         if( last_non_zero_element <= row_end )
            last_non_zero_element = row_end + 1;
         rows_info[ row ]. last ++;
         dbgExpr( last_non_zero_element );
         return true; 
      }

      dbgExpr( "No free space in row " << row );
      // there is no space for adding the new element into the row
      // we must shift all data behind up to the last-non-zero-element

      // first check if there is enough allocated memory for shifting
      
      if( rows_info[ size ]. last + 1 >= allocated_elements &&
         ! AllocateNewMemory( 1 ) ) return false;

      long int j = last_non_zero_element ++;
#ifdef CSR_MATRIX_TUNING
         data_shifts += j - i;
#endif
      while( j > i )
      {
         data[ j ] = data[ j - 1 ]; // data[ j ] = data[ -- j ]; really does not work on gcc 4.1.2
         j --;
      }
      data[ i ]. column = col;
      data[ i ]. value = v;
      if( col < row && rows_info[ row ]. diagonal != -1 ) rows_info[ row ]. diagonal ++;
      if( row == col )
      {
         dbgCout( "Setting pointer to diagonal element on row " << row << " to " << i );
         rows_info[ row ]. diagonal = i;
      }
      rows_info[ row ]. last ++;
      j = row + 1;
#ifdef CSR_MATRIX_TUNING
         data_shifts += size - j + 1;
#endif
      while( j <= size )
      {
         rows_info[ j ]. first ++;
         if( rows_info[ j ]. diagonal != -1 ) rows_info[ j ]. diagonal ++;
         rows_info[ j ]. last ++;
         
         assert( rows_info[ j ]. diagonal == -1 ||
          ( rows_info[ j ]. first <= rows_info[ j ]. diagonal &&
            rows_info[ j ]. diagonal < rows_info[ j ]. last ) );
         assert( rows_info[ j ]. diagonal == - 1 ||
          ( data[ rows_info[ j ]. diagonal ]. column == j ) );
         j ++;
      }
      return true;
   };

   //! Shifts data from the given position (@param from) by @param shift elements.
   /*! It also manages rof boundaries, diagonal entry pointers and it allocates 
       new memory if it is neccesary.
    */
   //m_bool ShiftData( m_int from, m_int shift );
   
   //! Dimension
   long int size;
   
   //! Array with matrix elements (AA and JA arrays)
   tnlCSRMatrixElement< T >* data;

   //! Number of allocated elements
   /*! It says how many elements are there in @param data array.
       If we need more additional elements are going to be allocated
       @see allocation_segement_size, @see AllocateNewMemory
    */
   long int allocated_elements;
   
   //! Segment size
   /*! It says by what amount the @param data array is goind to be
       increased during AllocateNewMemory call.
    */
   long int allocation_segment_size;
  
   //! This points right behind the last non-zero element in the @param data array
   /*! It is important for shifting data in @param data array when a new element
       is inserted
    */
   long int last_non_zero_element;

   //! Field of row info structures - one for each row
   /*! To make algortithms easier we have info even for the row number size + 1.
    */
   tnlCSRMatrixRowInfo* rows_info;


#ifdef CSR_MATRIX_TUNING
   long int data_shifts;
   long int data_seeks;
   long int re_allocs;

   public:

   void PrintStatistics();
   void ResetStatistics();
#endif

#ifdef DEBUG
   private:
   void PrintDataArray() const
   {
      long int i;
      for( i = 0; i < last_non_zero_element; i ++ )
         if( data[ i ]. value != 0.0 )
            cout << i << " col. " << data[ i ]. column << " val. " << data[ i ]. value << endl;
   };
#endif

};

template< typename T > ostream& operator << ( ostream& o_str, const tnlCSRMatrix< T >& A )
{
   long int size = A. GetSize();
   long int i, j;
   o_str << endl;
   for( i = 0; i < size; i ++ )
   {
      for( j = 0; j < size; j ++ )
      {
         const T& v = A. GetElement( i, j );
         if( v == 0.0 ) o_str << setw( 12 ) << ".";
         else o_str << setprecision( 6 ) << setw( 12 ) << v;
      }
      o_str << endl;
   }
   return o_str;
}

#endif
