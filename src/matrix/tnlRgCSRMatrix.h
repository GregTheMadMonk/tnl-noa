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

   Index getGroupSize( const Index groupId ) const;

   Index getCUDABlockSize() const;

   //! This can only be a multiple of the groupSize
   void setCUDABlockSize( Index blockSize );

   //! Sets the number of row and columns.
   bool setSize( Index new_size );

   //! Allocate memory for the nonzero elements.
   bool setNonzeroElements( Index elements );

   void reset();

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

   /****
    * This method sets parameters of the format.
    * If it is called after method copyFrom, the matrix will be broken.
    * TODO: Add state ensuring that this situation will lead to error.
    */
   void tuneFormat( const Index groupSize,
                    const bool useAdaptiveGroupSize = false,
                    const tnlAdaptiveGroupSizeStrategy adaptiveGroupSizeStrategy = tnlAdaptiveGroupSizeStrategyByAverageRowSize );

   bool copyFrom( const tnlCSRMatrix< Real, tnlHost, Index >& csr_matrix );


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
                  const tnlString& format = tnlString( "" ),
		            const Index lines = 0 ) const;

   bool draw( ostream& str,
              const tnlString& format,
              tnlCSRMatrix< Real, Device, Index >* csrMatrix = 0,
              int verbose = 0 );

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

   tnlAdaptiveGroupSizeStrategy adaptiveGroupSizeStrategy;

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

template< class Real, typename Index, bool useCache >
__global__ void sparseOldCSRMatrixVectorProductKernel( Index size,
                                                       Index block_size,
                                                       const Real* nonzeroElements,
                                                       const Index* columns,
                                                       const Index* groupOffsets,
                                                       const Index* nonzeroElementsInRow,
                                                       const Real* vec_x,
                                                       Real* vec_b );

template< class Real, typename Index >
__global__ void tnlRgCSRMatrixVectorProductKernel( Index size,
                                                   Index groupSize,
                                                   const Real* nonzeroElements,
                                                   const Index* columns,
                                                   const Index* groupOffsets,
                                                   const Index* nonzerosInRow,
                                                   const Real* vec_x,
                                                   Real* vec_b );

template< class Real, typename Index >
__global__ void tnlRgCSRMatrixAdpativeGroupSizeVectorProductKernel( Index size,
                                                                    const Index* groupSize,
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
  adaptiveGroupSizeStrategy( tnlAdaptiveGroupSizeStrategyByAverageRowSize ),
  nonzeroElements( name + " : nonzeroElements" ),
  columns( name + " : columns" ),
  groupOffsets( name + " : block-offsets" ),
  nonzeroElementsInRow( name + " : nonzerosInRow" ),
  groupSize( 16 ),
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
Index tnlRgCSRMatrix< Real, Device, Index > :: getGroupSize( const Index groupId ) const
{
   /****
    * If we use adaptive group sizes they are stored explicitly.
    */
   if( this -> useAdaptiveGroupSize )
      return this -> adaptiveGroupSizes. getElement( groupId );
   /****
    * The last group may be smaller even if we have constant group size.
    */
   if( groupId * this -> groupSize > this -> getSize() )
      return this -> getSize() % this -> groupSize;
   /***
    * If it is not the last group, return the common group size.
    */
   return this -> groupSize;
}

template< typename Real, tnlDevice Device, typename Index >
Index tnlRgCSRMatrix< Real, Device, Index > :: getCUDABlockSize() const
{
   return cudaBlockSize;
}

template< typename Real, tnlDevice Device, typename Index >
void tnlRgCSRMatrix< Real, Device, Index > :: setCUDABlockSize( Index blockSize )
{
   tnlAssert( blockSize % this -> groupSize == 0, )
   cudaBlockSize = blockSize;
}

template< typename Real, tnlDevice Device, typename Index >
bool tnlRgCSRMatrix< Real, Device, Index > :: setSize( Index new_size )
{
   this -> size = new_size;
   if( ! groupOffsets. setSize( this -> getSize() / groupSize + ( this -> getSize() % groupSize != 0 ) + 1 ) ||
	    ! nonzeroElementsInRow. setSize( this -> getSize() ) ||
	    ! adaptiveGroupSizes. setSize( this -> getSize() + 1 ) )
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
void tnlRgCSRMatrix< Real, Device, Index > :: reset()
{
   this -> size = 0;
   nonzeroElements. reset();
   columns. reset();
   groupOffsets. reset();
   nonzeroElementsInRow. reset();
   adaptiveGroupSizes. reset();
   useAdaptiveGroupSize = false;
   last_nonzero_element = 0;
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
void tnlRgCSRMatrix< Real, Device, Index > :: tuneFormat( const Index groupSize,
                                                          const bool useAdaptiveGroupSize,
                                                          const tnlAdaptiveGroupSizeStrategy adaptiveGroupSizeStrategy )
{
   tnlAssert( this -> groupSize > 0, );
   this -> groupSize = groupSize;
   this -> useAdaptiveGroupSize = useAdaptiveGroupSize;
   this -> adaptiveGroupSizeStrategy = adaptiveGroupSizeStrategy;
}


template< typename Real, tnlDevice Device, typename Index >
bool tnlRgCSRMatrix< Real, Device, Index > :: copyFrom( const tnlCSRMatrix< Real, tnlHost, Index >& csr_matrix )
{
	dbgFunctionName( "tnlRgCSRMatrix< Real, tnlHost >", "copyFrom" );

	if( ! this -> setSize( csr_matrix. getSize() ) )
		return false;
	dbgExpr( csr_matrix. getSize() );

	/****
	 * In case of adaptive group sizes compute maximum number of the non-zero elements in group.
	 */
	Index maxNonzeroElementsInGroup( 0 );
	if( this -> useAdaptiveGroupSize )
	{
	   if( this -> adaptiveGroupSizeStrategy == tnlAdaptiveGroupSizeStrategyByAverageRowSize )
	   {
	      const Index averageRowSize = ceil( ( float ) csr_matrix. getNonzeroElements() / ( float ) csr_matrix. getSize() );
	      maxNonzeroElementsInGroup = averageRowSize * groupSize;
	   }
	   if( this -> adaptiveGroupSizeStrategy == tnlAdaptiveGroupSizeStrategyByFirstGroup )
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
		   if( ! this -> useAdaptiveGroupSize || nonzeroElementsInGroup > maxNonzeroElementsInGroup )
		   {
		      if( this -> useAdaptiveGroupSize )
		         adaptiveGroupSizes[ numberOfGroups + 1 ] = currentGroupSize;

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
   {
       adaptiveGroupSizes[ numberOfGroups + 1 ] = currentGroupSize;

       /****
        * Compute prefix sum on group sizes
        */
       for( Index i = 1; i < adaptiveGroupSizes; i ++ )
          adaptiveGroupSizes[ i ] += adaptiveGroupSizes[ i - 1 ];
   }
	dbgCout( numberOfGroups << "-th group size is " << currentGroupSize );
	numberOfGroups ++;


	/****
	 * Allocate the non-zero elements (they contains some artificial zeros.)
	 */
	dbgCout( "Allocating " << Max( 1, total_elements ) << " elements.");
	if( ! setNonzeroElements( Max( 1, total_elements ) ) )
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
             * We start with the offset k within the group and
             * we insert the data with a stride equal to the group size.
             * j - is the element position in the nonzeroElements in this matrix
             */
            Index j = groupOffsets[ groupId ] + k;                   // position of the first element of the row

            /****
             * Get the element position
             */
            tnlAssert( elementRow < this -> getSize(), cerr << "Element row = " << elementRow );
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

   /****
    * TODO: add variable group size support
    */
   groupSize = rgCSRMatrix. groupSize;
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
   adaptiveGroupSizes = rgCSRMatrix. adaptiveGroupSizes;
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
#ifdef UNDEF
      /****
       * This is exact emulation of the CUDA kernel
       */
      Index blockSize = 256; //this -> getCUDABlockSize();
      const Index size = this -> getSize();
      Index gridSize = size / blockSize + ( size % blockSize != 0 ) + 1;
      for( Index blockIdx = 0; blockIdx < gridSize; blockIdx ++ )
         for( Index threadIdx = 0; threadIdx < blockSize; threadIdx ++ )
         {
            const Index rowIndex = blockIdx * blockSize + threadIdx;
            if( rowIndex >= size )
               continue;

            const Index groupIndex = floor( threadIdx / this -> groupSize );
            const Index globalGroupIndex = blockIdx * ( blockSize / groupSize ) + groupIndex;
            const Index rowOffsetInGroup = rowIndex % this -> groupSize ;

            /****
             * The last block may be smaller then the global block_size.
             * We store it in the current_block_size
             */
            Index currentGroupSize = this -> groupSize;
            if( ( globalGroupIndex + 1 ) * this -> groupSize > size )
               currentGroupSize =  size % this -> groupSize;

            Real product( 0.0 );
            Index pos = groupOffsets[ globalGroupIndex ] + rowOffsetInGroup;
            const Index nonzeros = nonzeroElementsInRow[ rowIndex ];

            for( Index i = 0; i < nonzeros; i ++ )
            {
               //if( columns[ pos ] < 0 || columns[ pos ] >= size )
               //   printf( "rowIndex = %d currentGroupSize = %d groupSize = %d columns[ pos ] = %d\n", rowIndex, currentGroupSize, groupSize, columns[ pos ] );
               Index column = columns[ pos ];
               if( column != -1 )
                  product += nonzeroElements[ pos ] * vec[ column ];
               else
                  cerr << "row = " << rowIndex
                       << " globalGroupId = " << globalGroupIndex
                       << " i = " << i
                       << " group size = " << currentGroupSize
                       << " offset = " << pos
                       << " nonzeros = " <<  nonzeros << endl;
               pos += currentGroupSize;
            }
            //printf( "rowIndex = %d \n", rowIndex );
            //Real* x = const_cast< Real* >( vec_x );
            result[ rowIndex ] = product;
         }
#endif
//#ifdef UNDEF
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
         Index first_row = groupId * groupSize;
         if( useAdaptiveGroupSize )
            first_row = adaptiveGroupSizes. getElement( groupId );

         Index csr_col = 0;
         Index row = firstRowOfGroup;
         for( Index block_row = 0; block_row < currentGroupSize; block_row ++ )
         {
            //const Index row = first_row + block_row;
            result[ row ] = 0.0;
            if( csr_col < row_lengths[ row ] )
            {
               result[ row ] += values[ block_begining ] * vec[ cols[ block_begining ] ];
            }
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
               {
                  result[ row ] += values[ block_begining ] * vec[ cols[ block_begining ] ];
               }
               block_begining ++;
               row ++;
            }
         }
         firstRowOfGroup += currentGroupSize;
      }
//#endif
   }
   if( Device == tnlCuda )
   {
#ifdef HAVE_CUDA
      Index blockSize = this -> getCUDABlockSize();
      const Index size = this -> getSize();

      /****
       * The following works only with constant group size
       */
      int gridSize = size / blockSize + ( size % blockSize != 0 ) + 1;
      dim3 gridDim( gridSize ), blockDim( blockSize );

      if( this -> useAdaptiveGroupSize )
         tnlRgCSRMatrixAdpativeGroupSizeVectorProductKernel( size,
                                                             adaptiveGroupSizes. getVector(),
                                                             nonzeroElements. getVector(),
                                                             columns. getVector(),
                                                             groupOffsets. getVector(),
                                                             nonzeroElementsInRow. getVector(),
                                                             vec. getVector(),
                                                             result. getVector() );
      else
         sparseRgCSRMatrixVectorProductKernel< Real, Index ><<< gridDim, blockDim >>>( size,
                                                                                       this -> groupSize,
                                                                                       nonzeroElements. getVector(),
                                                                                       columns. getVector(),
                                                                                       groupOffsets. getVector(),
                                                                                       nonzeroElementsInRow. getVector(),
                                                                                       vec. getVector(),
                                                                                       result. getVector() );
       cudaThreadSynchronize();
       CHECK_CUDA_ERROR;
#else
       cerr << "CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif
   }

}

template< typename Real, tnlDevice Device, typename Index >
void tnlRgCSRMatrix< Real, Device, Index > :: printOut( ostream& str,
                                                        const tnlString& format,
		                                                  const Index lines ) const
{
   if( format == "" || format == "text" )
   {
      str << "Structure of tnlRgCSRMatrix" << endl;
      str << "Matrix name:" << this -> getName() << endl;
      str << "Matrix size:" << this -> getSize() << endl;
      str << "Allocated elements:" << nonzeroElements. getSize() << endl;
      str << "Number of groups: " << groupOffsets. getSize() << endl;

      Index print_lines = lines;
      if( ! print_lines )
         print_lines = this -> getSize();

      for( Index i = 0; i < this -> numberOfGroups; i ++ )
      {
         if( i * groupSize > print_lines )
            return;
         str << endl << "Block number: " << i << endl;
         str << " Group size: " << this -> getGroupSize( i ) << endl;
         str << " Group non-zeros: ";
         for( Index k = i * groupSize; k < ( i + 1 ) * groupSize && k < this -> getSize(); k ++ )
            str << nonzeroElementsInRow. getElement( k ) << "  ";
         str << endl;
         str << " Group data: "
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
         str << endl << "Offsets: ";
         for( Index k = groupOffsets. getElement( i );
              k < groupOffsets. getElement( i + 1 );
              k ++ )
            str << setprecision( 5 ) << setw( 8 )
                << k << " ";

      }
      str << endl;
      return;
   }
   if( format == "html" )
   {
      str << "<h1>Structure of tnlRgCSRMatrix</h1>" << endl;
      str << "<b>Matrix name:</b> " << this -> getName() << "<p>" << endl;
      str << "<b>Matrix size:</b> " << this -> getSize() << "<p>" << endl;
      str << "<b>Allocated elements:</b> " << nonzeroElements. getSize() << "<p>" << endl;
      str << "<b>Number of groups:</b> " << this -> numberOfGroups << "<p>" << endl;
      str << "<table border=1>" << endl;
      str << "<tr> <td> <b> GroupId </b> </td> <td> <b> Size </b> </td> <td> <b> % of nonzeros </b> </td> </tr>" << endl;
      Index print_lines = lines;
      if( ! print_lines )
         print_lines = this -> getSize();

      for( Index i = 0; i < this -> numberOfGroups; i ++ )
      {
         if( i * groupSize > print_lines )
            return;
         double filling = ( double ) ( this -> groupOffsets. getElement( i + 1 ) - this -> groupOffsets. getElement( i ) ) /
                          ( double ) this -> nonzeroElements. getSize();
         str << "<tr> <td> " << i << "</td> <td>" << this -> getGroupSize( i ) << " </td> <td> " << 100.0 * filling << "% </td></tr>" << endl;
      }
      str << "</table>" << endl;
      str << endl;
   }
};

template< typename Real, tnlDevice Device, typename Index >
bool tnlRgCSRMatrix< Real, Device, Index > :: draw( ostream& str,
                                                    const tnlString& format,
                                                    tnlCSRMatrix< Real, Device, Index >* csrMatrix,
                                                    int verbose )
{
   if( Device == tnlCuda )
   {
      cerr << "Drawing of matrices stored on the GPU is not supported yet." << endl;
      return false;
   }
   if( format == "gnuplot" )
      return tnlMatrix< Real, Device, Index > ::  draw( str, format, csrMatrix, verbose );
   if( format == "eps" )
   {
      const int elementSize = 10;
      this -> writePostscriptHeader( str, elementSize );

      /****
       * Draw the groups
       */
      for( Index groupId = 0; groupId < numberOfGroups; groupId ++ )
      {
         const Index groupSize = getCurrentGroupSize( groupId );
         if( groupId % 2 == 0 )
            str << "0.9 0.9 0.9 setrgbcolor" << endl;
         else
            str << "0.8 0.8 0.8 setrgbcolor" << endl;
         str << "0 -" << groupSize * elementSize
             << " translate newpath 0 0 " << this -> getSize() * elementSize
             << " " << groupSize * elementSize << " rectfill" << endl;
      }
      /****
       * Restore black color and the origin of the coordinates
       */
      str << "0 0 0 setrgbcolor" << endl;
      str << "0 " << this -> getSize() * elementSize << " translate" << endl;

      if( csrMatrix )
         csrMatrix -> writePostscriptBody( str, elementSize, verbose );
      else
         this -> writePostscriptBody( str, elementSize, verbose );

      str << "showpage" << endl;
      str << "%%EOF" << endl;

      if( verbose )
         cout << endl;
      return true;
   }
}

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
      product += nonzeroElements[ pos ] * vec_x[ columns[ pos ] ];
      pos += current_block_size;
   }
   vec_b[ row ] = product;
}


template< class Real, typename Index >
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
    * This kernel assumes that all groups are equal sized. One block
    * can process more groups and there is always one thread mapped to one row.
    */
   const Index rowIndex = blockIdx. x * blockDim. x + threadIdx. x;
   if( rowIndex >= size )
      return;

   const Index groupIndex = threadIdx. x / groupSize ;
   const Index rowOffsetInGroup = rowIndex % groupSize ;
   const Index globalGroupIndex = rowIndex / groupSize;

   /****
    * The last block may be smaller then the global block_size.
    * We store it in the current_block_size
    */
   Index currentGroupSize = groupSize;
   if( ( globalGroupIndex + 1 ) * groupSize > size )
      currentGroupSize =  size % groupSize;

   Real product( 0.0 );
   Index pos = groupOffsets[ globalGroupIndex ] + rowOffsetInGroup;
   const Index nonzeros = nonzerosInRow[ rowIndex ];
   for( Index i = 0; i < nonzeros; i ++ )
   {
         product += nonzeroElements[ pos ] * vec_x[ columns[ pos ] ];
      pos += currentGroupSize;
   }
   vec_b[ rowIndex ] = product;
}

template< class Real, typename Index >
__global__ void tnlRgCSRMatrixAdpativeGroupSizeVectorProductKernel( Index size,
                                                                    const Index* groupSize,
                                                                    const Real* nonzeroElements,
                                                                    const Index* columns,
                                                                    const Index* groupOffsets,
                                                                    const Index* nonzerosInRow,
                                                                    const Real* vec_x,
                                                                    Real* vec_b )
{
   /****
    * Now the group size is variable and one group is processed
    * by one block.
    */
   const Index rowIndex = blockIdx. x * blockDim. x + threadIdx. x;
   if( rowIndex >= size )
      return;

   const Index groupIndex = threadIdx. x / groupSize ;
   const Index rowOffsetInGroup = rowIndex % groupSize ;
   const Index globalGroupIndex = rowIndex / groupSize;

   /****
    * The last block may be smaller then the global block_size.
    * We store it in the current_block_size
    */
   Index currentGroupSize = groupSize;
   if( ( globalGroupIndex + 1 ) * groupSize > size )
      currentGroupSize =  size % groupSize;

   Real product( 0.0 );
   Index pos = groupOffsets[ globalGroupIndex ] + rowOffsetInGroup;
   const Index nonzeros = nonzerosInRow[ rowIndex ];
   for( Index i = 0; i < nonzeros; i ++ )
   {
         product += nonzeroElements[ pos ] * vec_x[ columns[ pos ] ];
      pos += currentGroupSize;
   }
   vec_b[ rowIndex ] = product;
}



#endif // ifdef HAVE_CUDA


#endif /* TNLRgCSRMATRIX_H_ */
