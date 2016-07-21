/***************************************************************************
                          tnlRgCSRMatrix.h  -  description
                             -------------------
    begin                : Jul 10, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


#ifndef TNLRGCSRMATRIX_H
#define TNLRGCSRMATRIX_H

#include <iostream>
#include <iomanip>
#include <TNL/Vectors/Vector.h>
#include <TNL/Assert.h>
#include <TNL/core/mfuncs.h>
#include <TNL/matrices/tnlMatrix.h>
#include <TNL/matrices/tnlCSRMatrix.h>
#include <TNL/debug/tnlDebug.h>

using namespace std;

enum tnlAdaptiveGroupSizeStrategy { tnlAdaptiveGroupSizeStrategyByAverageRowSize,
                                    tnlAdaptiveGroupSizeStrategyByFirstGroup };

//! Matrix storing the non-zero elements in the Row-grouped CSR (Compressed Sparse Row) format
/*!
 */
template< typename Real, typename Device = tnlHost, typename Index = int  >
class tnlRgCSRMatrix : public tnlMatrix< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   //! Basic constructor
   tnlRgCSRMatrix( const String& name );

   const String& getMatrixClass() const;

   String getType() const;

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


   template< typename Device2 >
   bool copyFrom( const tnlRgCSRMatrix< Real, Device2, Index >& rgCSRMatrix );

   Real getElement( Index row,
                    Index column ) const;

   Real rowProduct( Index row,
                    const tnlVector< Real, Device, Index >& vector ) const;

   void vectorProduct( const tnlVector< Real, Device, Index >& x,
                       tnlVector< Real, Device, Index >& b ) const;

   Real getRowL1Norm( Index row ) const
   { abort(); };

   void multiplyRow( Index row, const Real& value )
   { abort(); };

   //! Prints out the matrix structure
   void printOut( std::ostream& str,
                  const String& format = String( "" ),
		            const Index lines = 0 ) const;

   bool draw( std::ostream& str,
              const String& format,
              tnlCSRMatrix< Real, Device, Index >* csrMatrix = 0,
              int verbose = 0 );

   protected:

   Index getGroupIndexFromRow( const Index row ) const;

   Index getRowIndexInGroup( const Index row, const Index groupId ) const;

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

   tnlVector< Real, Device, Index > nonzeroElements;

   tnlVector< Index, Device, Index > columns;

   tnlVector< Index, Device, Index > groupOffsets;

   tnlVector< Index, Device, Index > nonzeroElementsInRow;

   Index groupSize;

   /****
    * This vector is only used if useAdaptiveGroupSize is true.
    */
   tnlVector< Index, Device, Index > adaptiveGroupSizes;

   /****
    * This variable is only used if useAdaptiveGroupSize is true.
    */
   Index numberOfGroups;

   Index cudaBlockSize;

   //int maxCudaGridSize;

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
__global__ void tnlRgCSRMatrixVectorProductKernel( const Index gridNumber,
                                                   const Index maxGridSize,
                                                   Index size,
                                                   Index groupSize,
                                                   const Real* nonzeroElements,
                                                   const Index* columns,
                                                   const Index* groupOffsets,
                                                   const Index* nonzerosInRow,
                                                   const Real* vec_x,
                                                   Real* vec_b );

template< class Real, typename Index >
__global__ void tnlRgCSRMatrixAdpativeGroupSizeVectorProductKernel( const Index gridNumber,
                                                                    const Index maxGridSize,
                                                                    Index size,
                                                                    const Index* groupSize,
                                                                    const Real* nonzeroElements,
                                                                    const Index* columns,
                                                                    const Index* groupOffsets,
                                                                    const Index* nonzerosInRow,
                                                                    const Real* vec_x,
                                                                    Real* vec_b );
#endif


template< typename Real, typename Device, typename Index >
tnlRgCSRMatrix< Real, Device, Index > :: tnlRgCSRMatrix( const String& name )
: tnlMatrix< Real, Device, Index >( name ),
  useAdaptiveGroupSize( false ),
  adaptiveGroupSizeStrategy( tnlAdaptiveGroupSizeStrategyByAverageRowSize ),
  nonzeroElements( String( name ) + ":nonzeroElements" ),
  columns( String( name ) + ":columns" ),
  groupOffsets( String( name ) + ":groupOffsets" ),
  nonzeroElementsInRow( String( name ) + ":nonzerosInRow" ),
  groupSize( 16 ),
  adaptiveGroupSizes( String( name ) + ":adaptiveGroupSizes" ),
  numberOfGroups( 0 ),
  cudaBlockSize( 256 ),
  artificial_zeros( 0 ),
  last_nonzero_element( 0 )
{
#ifdef HAVE_CUDA
   int cudaDevice;
   cudaGetDevice( &cudaDevice );
   cudaDeviceProp deviceProperties;
   cudaGetDeviceProperties( &deviceProperties, cudaDevice );
   //this->maxCudaGridSize = deviceProperties. maxGridSize[ 0 ];
#endif
};

template< typename Real, typename Device, typename Index >
const String& tnlRgCSRMatrix< Real, Device, Index > :: getMatrixClass() const
{
   return tnlMatrixClass :: main;
};

template< typename Real, typename Device, typename Index >
String tnlRgCSRMatrix< Real, Device, Index > :: getType() const
{
   return String( "tnlRgCSRMatrix< ") +
          String( getType( Real( 0.0 ) ) ) +
          String( ", " ) +
          Device :: getDeviceType() +
          String( ", " ) +
          getType( Index( 0 ) ) +
          String( " >" );
   // TODO: add value of useAdaptiveGroupSize
};

template< typename Real, typename Device, typename Index >
Index tnlRgCSRMatrix< Real, Device, Index > :: getCUDABlockSize() const
{
   return cudaBlockSize;
}

template< typename Real, typename Device, typename Index >
void tnlRgCSRMatrix< Real, Device, Index > :: setCUDABlockSize( Index blockSize )
{
   Assert( blockSize % this->groupSize == 0, )
   cudaBlockSize = blockSize;
}

template< typename Real, typename Device, typename Index >
bool tnlRgCSRMatrix< Real, Device, Index > :: setSize( Index new_size )
{
   this->size = new_size;
   if( ! groupOffsets. setSize( this->getSize() / groupSize + ( this->getSize() % groupSize != 0 ) + 1 ) ||
	    ! nonzeroElementsInRow. setSize( this->getSize() ) ||
	    ! adaptiveGroupSizes. setSize( this->getSize() + 1 ) )
      return false;
   groupOffsets. setValue( 0 );
   nonzeroElementsInRow. setValue( 0 );
   adaptiveGroupSizes. setValue( 0 );
   last_nonzero_element = 0;
   return true;
};

template< typename Real, typename Device, typename Index >
bool tnlRgCSRMatrix< Real, Device, Index > :: setNonzeroElements( Index elements )
{
   if( ! nonzeroElements. setSize( elements ) ||
	    ! columns. setSize( elements ) )
      return false;
   nonzeroElements. setValue( 0.0 );
   columns. setValue( -1 );
   return true;
};

template< typename Real, typename Device, typename Index >
void tnlRgCSRMatrix< Real, Device, Index > :: reset()
{
   this->size = 0;
   nonzeroElements. reset();
   columns. reset();
   groupOffsets. reset();
   nonzeroElementsInRow. reset();
   adaptiveGroupSizes. reset();
   useAdaptiveGroupSize = false;
   last_nonzero_element = 0;
};


template< typename Real, typename Device, typename Index >
Index tnlRgCSRMatrix< Real, Device, Index > :: getNonzeroElements() const
{
   Assert( nonzeroElements. getSize() > artificial_zeros, );
	return nonzeroElements. getSize() - artificial_zeros;
}

template< typename Real, typename Device, typename Index >
Index tnlRgCSRMatrix< Real, Device, Index > :: getArtificialZeroElements() const
{
	return artificial_zeros;
}

template< typename Real, typename Device, typename Index >
void tnlRgCSRMatrix< Real, Device, Index > :: tuneFormat( const Index groupSize,
                                                          const bool useAdaptiveGroupSize,
                                                          const tnlAdaptiveGroupSizeStrategy adaptiveGroupSizeStrategy )
{
   Assert( this->groupSize > 0, );
   this->groupSize = groupSize;
   this->useAdaptiveGroupSize = useAdaptiveGroupSize;
   this->adaptiveGroupSizeStrategy = adaptiveGroupSizeStrategy;
}


template< typename Real, typename Device, typename Index >
bool tnlRgCSRMatrix< Real, Device, Index > :: copyFrom( const tnlCSRMatrix< Real, tnlHost, Index >& csr_matrix )
{
	dbgFunctionName( "tnlRgCSRMatrix< Real, tnlHost >", "copyFrom" );

   if( Device :: getDevice() == tnlCudaDevice )
   {
      Assert( false,
                 std::cerr << "Conversion from tnlCSRMatrix on the host to the tnlRgCSRMatrix on the CUDA device is not implemented yet."; );
      //TODO: implement this
      return false;
   }

	if( ! this->setSize( csr_matrix. getRows() ) )
		return false;
	dbgExpr( csr_matrix. getSize() );

	/****
	 * In case of adaptive group sizes compute maximum number of the non-zero elements in group.
	 */
	Index maxNonzeroElementsInGroup( 0 );
	if( this->useAdaptiveGroupSize )
	{
	   if( this->adaptiveGroupSizeStrategy == tnlAdaptiveGroupSizeStrategyByAverageRowSize )
	   {
	      const Index averageRowSize = ceil( ( double ) csr_matrix. getNumberOfAllocatedElements() / ( double ) csr_matrix. getRows() );
	      maxNonzeroElementsInGroup = averageRowSize * groupSize;
	   }
	   //if( this->adaptiveGroupSizeStrategy == tnlAdaptiveGroupSizeStrategyByFirstGroup )
	   //   for( Index row = 0; row < groupSize; row ++ )
	   //      maxNonzeroElementsInGroup += csr_matrix. getNonzeroElementsInRow( row );
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
	for( Index i = 0; i < this->getSize(); i ++ )
	{
		if( i > 0 && i % groupSize == 0 )
		{
		   currentGroupSize += groupSize;
		   if( ! this->useAdaptiveGroupSize ||
		       ( nonzeroElementsInGroup > maxNonzeroElementsInGroup && cudaBlockSize % currentGroupSize == 0 )
		       || currentGroupSize == cudaBlockSize )
		   {
		      if( this->useAdaptiveGroupSize )
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
		max_row_in_block = max( max_row_in_block, nonzeroElementsInRow[ i ] );

	}
	dbgExpr( processedLines );
	currentGroupSize = this->getSize() - processedLines;
	total_elements += max_row_in_block * ( currentGroupSize );
	groupOffsets[ numberOfGroups + 1 ] = total_elements;
   if( useAdaptiveGroupSize )
   {
       adaptiveGroupSizes[ numberOfGroups + 1 ] = currentGroupSize;

       /****
        * Compute prefix sum on group sizes
        */
       for( Index i = 1; i < adaptiveGroupSizes. getSize(); i ++ )
          adaptiveGroupSizes[ i ] += adaptiveGroupSizes[ i - 1 ];
       dbgExpr( adaptiveGroupSizes );
       dbgExpr( this->getSize() );
   }
	dbgCout( numberOfGroups << "-th group size is " << currentGroupSize );
	numberOfGroups ++;


	/****
	 * Allocate the non-zero elements (they contain some artificial zeros.)
	 */
	dbgCout( "Allocating " << max( 1, total_elements ) << " elements.");
	if( ! setNonzeroElements( max( 1, total_elements ) ) )
		return false;
	artificial_zeros = total_elements - csr_matrix. getNonzeroElements();

	dbgCout( "Inserting data " );
	if( Device :: getDevice() == tnlHostDevice )
	{
	   Index elementRow( 0 );
      /***
       * Insert the data into the groups.
       * We go through the groups.
       */
      for( Index groupId = 0; groupId < numberOfGroups; groupId ++ )
      {
         Index currentGroupSize = getCurrentGroupSize( groupId );
         dbgCout( "GroupId = " << groupId << "/" << numberOfGroups );
         dbgExpr( groupOffsets[ groupId ] );
         dbgExpr( currentGroupSize );

         /****
          * We insert 'currentGroupSize' rows in this matrix with the stride
          * given by the same number.
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
            Assert( elementRow < this->getSize(), std::cerr << "Element row = " << elementRow );
            Index elementPos = csr_matrix. row_offsets[ elementRow ];
            while( elementPos < csr_matrix. row_offsets[ elementRow + 1 ] )
            {
               dbgCoutLine( "Inserting on position " << j
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
	return true;
};

template< typename Real, typename Device, typename Index >
   template< typename Device2 >
bool
tnlRgCSRMatrix< Real, Device, Index >::
copyFrom( const tnlRgCSRMatrix< Real, Device2, Index >& rgCSRMatrix )
{
   dbgFunctionName( "tnlRgCSRMatrix< Real, Device, Index >", "copyFrom" );
   Assert( rgCSRMatrix. getSize() > 0, std::cerr << "Copying from matrix with non-positiove size." );

   this->cudaBlockSize = rgCSRMatrix. cudaBlockSize;
   this->groupSize = rgCSRMatrix. groupSize;
   if( ! this->setSize( rgCSRMatrix. getSize() ) )
      return false;

   /****
    * Allocate the non-zero elements (they contains some artificial zeros.)
    */
   Index total_elements = rgCSRMatrix. getNonzeroElements() +
                          rgCSRMatrix. getArtificialZeroElements();
   dbgCout( "Allocating " << total_elements << " elements.");
   if( ! setNonzeroElements( total_elements ) )
      return false;
   this->artificial_zeros = total_elements - rgCSRMatrix. getNonzeroElements();

   this->nonzeroElements = rgCSRMatrix. nonzeroElements;
   this->columns = rgCSRMatrix. columns;
   this->groupOffsets = rgCSRMatrix. groupOffsets;
   this->nonzeroElementsInRow = rgCSRMatrix. nonzeroElementsInRow;
   this->last_nonzero_element = rgCSRMatrix. last_nonzero_element;

   this->numberOfGroups = rgCSRMatrix. numberOfGroups;
   this->adaptiveGroupSizes = rgCSRMatrix. adaptiveGroupSizes;
   this->useAdaptiveGroupSize = rgCSRMatrix. useAdaptiveGroupSize;
   this->adaptiveGroupSizeStrategy = rgCSRMatrix. adaptiveGroupSizeStrategy;
   //this->maxCudaGridSize = rgCSRMatrix. maxCudaGridSize;
   return true;
};


template< typename Real, typename Device, typename Index >
Real tnlRgCSRMatrix< Real, Device, Index > :: getElement( Index row,
                                                          Index column ) const
{
   dbgFunctionName( "tnlRgCSRMatrix< Real, Device, Index >", "getElement" );
	Assert( 0 <= row && row < this->getSize(),
			   std::cerr << "The row is outside the matrix." );
   if( Device :: getDevice() == tnlHostDevice )
   {
      Index groupId = getGroupIndexFromRow( row );
      Index groupRow = getRowIndexInGroup( row, groupId );
      Index groupOffset = groupOffsets[ groupId ];
      Index currentGroupSize = getCurrentGroupSize( groupId );

      dbgCout( "row = " << row
               << " groupId = " << groupId
               << " groupSize = " << currentGroupSize
               << " groupRow = " << groupRow
               << " groupOffset = " << groupOffset );

      Index pos = groupOffset + groupRow;
      for( Index i = 0; i < nonzeroElementsInRow[ row ]; i ++ )
      {
         if( columns[ pos ] == column )
            return nonzeroElements[ pos ];
         pos += currentGroupSize;
      }
      return Real( 0.0 );
   }
   if( Device :: getDevice() == tnlCudaDevice )
   {
      Assert( false,
                std::cerr << "tnlRgCSRMatrix< Real, tnlCuda, Index > ::getElement is not implemented yet." );
      //TODO: implement this

   }
}

template< typename Real, typename Device, typename Index >
Real tnlRgCSRMatrix< Real, Device, Index > :: rowProduct( Index row,
                                                          const tnlVector< Real, Device, Index >& vec ) const
{
   Assert( 0 <= row && row < this->getSize(),
              std::cerr << "The row is outside the matrix." );
   Assert( vec. getSize() == this->getSize(),
              std::cerr << "The matrix and vector for multiplication have different sizes. "
                   << "The matrix size is " << this->getSize() << "."
                   << "The vector size is " << vec. getSize() << std::endl; );

   if( Device :: getDevice() == tnlHostDevice )
   {
      Assert( false, );
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
         Assert( pos < nonzeroElements. getSize(), );
         product += nonzeroElements[ pos ] * vec[ columns[ pos ] ];
         pos += currentGroupSize;
      }
      return product;
   }
   if( Device :: getDevice() == tnlCudaDevice )
   {
      Assert( false,
               std::cerr << "tnlRgCSRMatrix< Real, tnlCuda > :: rowProduct is not implemented yet." );
      //TODO: implement this
   }
}

template< typename Real, typename Device, typename Index >
void tnlRgCSRMatrix< Real, Device, Index > :: vectorProduct( const tnlVector< Real, Device, Index >& vec,
                                                             tnlVector< Real, Device, Index >& result ) const
{
   dbgFunctionName( "tnlRgCSRMatrix< Real, tnlHost >", "vectorProduct" )
   Assert( vec. getSize() == this->getSize(),
              std::cerr << "The matrix and vector for a multiplication have different sizes. "
                   << "The matrix size is " << this->getSize() << "."
                   << "The vector size is " << vec. getSize() << std::endl; );
   Assert( result. getSize() == this->getSize(),
              std::cerr << "The matrix and result vector of a multiplication have different sizes. "
                   << "The matrix size is " << this->getSize() << "."
                   << "The vector size is " << vec. getSize() << std::endl; );

   if( Device :: getDevice() == tnlHostDevice )
   {
//#ifdef UNDEF
      /****
       * This is exact emulation of the CUDA kernel
       */
      Index blockSize = 256; //this->getCUDABlockSize();
      const Index size = this->getSize();
      Index gridSize = size / blockSize + ( size % blockSize != 0 ) + 1;
      for( Index blockIdx = 0; blockIdx < gridSize; blockIdx ++ )
         for( Index threadIdx = 0; threadIdx < blockSize; threadIdx ++ )
         {
            const Index rowIndex = blockIdx * blockSize + threadIdx;
            if( rowIndex >= size )
               continue;

            const Index groupIndex = floor( threadIdx / this->groupSize );
            const Index globalGroupIndex = this->getGroupIndexFromRow( rowIndex );
            const Index rowOffsetInGroup = this->getRowIndexInGroup( rowIndex, globalGroupIndex );
            const Index currentGroupSize = this->getCurrentGroupSize( globalGroupIndex );
            Assert( rowOffsetInGroup < currentGroupSize, std::cerr << "rowOffsetInGroup = " << rowOffsetInGroup << ", currentGroupSize = " << currentGroupSize  );

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
               /*else
                  std::cerr << "row = " << rowIndex
                       << " globalGroupId = " << globalGroupIndex << "/" << numberOfGroups - 1
                       << " i = " << i
                       << " group size = " << currentGroupSize
                       << " offset = " << pos
                       << " nonzeros = " <<  nonzeros << std::endl;*/
               pos += currentGroupSize;
            }
            //printf( "rowIndex = %d \n", rowIndex );
            //Real* x = const_cast< Real* >( vec_x );
            result[ rowIndex ] = product;
         }
//#endif
#ifdef UNDEF
      const Index blocks_num = groupOffsets. getSize() - 1;
      const Index* row_lengths = nonzeroElementsInRow. getData();
      const Real* values = nonzeroElements. getData();
      const Index* cols = columns. getData();
      const Index* groupOffset = groupOffsets. getData();
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
#endif
   }
   if( Device :: getDevice() == tnlCudaDevice )
   {
#ifdef HAVE_CUDA
      Index blockSize = this->getCUDABlockSize();
      const Index size = this->getSize();

      if( this->useAdaptiveGroupSize )
      {
         int gridSize = this->numberOfGroups;
         int numberOfGrids = gridSize / maxCudaGridSize + ( gridSize % maxCudaGridSize != 0 );
         int gridNumber( 0 );
         while( gridSize > 0 )
         {
            int currentGridSize = min( gridSize, maxCudaGridSize );
            //cerr << "Current grid size = " << currentGridSize << std::endl;
            dim3 gridDim( currentGridSize ), blockDim( blockSize );
            size_t sharedBytes = blockDim. x * sizeof( Real );
            /*tnlRgCSRMatrixAdpativeGroupSizeVectorProductKernel< Real, Index >
                                                              <<< gridDim, blockDim, sharedBytes >>>
                                                              ( gridNumber,
                                                                this->maxCudaGridSize,
                                                                size,
                                                                adaptiveGroupSizes. getData(),
                                                                nonzeroElements. getData(),
                                                                columns. getData(),
                                                                groupOffsets. getData(),
                                                                nonzeroElementsInRow. getData(),
                                                                vec. getData(),
                                                                result. getData() );*/
            gridSize -= currentGridSize;
            gridNumber ++;
         }
      }
      else
      {
         int gridSize = size / blockSize + ( size % blockSize != 0 ) + 1;
         int gridNumber( 0 );
         while( gridSize > 0 )
         {
            /*int currentGridSize = min( gridSize, this->maxCudaGridSize );
            dim3 gridDim( currentGridSize ), blockDim( blockSize );
            tnlRgCSRMatrixVectorProductKernel< Real, Index >
                                             <<< gridDim, blockDim >>>
                                             ( gridNumber,
                                               this->maxCudaGridSize,
                                               size,
                                               this->groupSize,
                                               nonzeroElements. getData(),
                                               columns. getData(),
                                               groupOffsets. getData(),
                                               nonzeroElementsInRow. getData(),
                                               vec. getData(),
                                               result. getData() );
            gridSize -= currentGridSize;
            gridNumber ++;*/
         }
      }
       cudaThreadSynchronize();
       checkCudaDevice;

#else
       tnlCudaSupportMissingMessage;;
#endif
   }

}

template< typename Real, typename Device, typename Index >
void tnlRgCSRMatrix< Real, Device, Index > :: printOut( std::ostream& str,
                                                        const String& name,
                                                        const String& format,
		                                                  const Index lines ) const
{
   if( format == "" || format == "text" )
   {
      str << "Structure of tnlRgCSRMatrix" << std::endl;
      str << "Matrix name:" << name << std::endl;
      str << "Matrix size:" << this->getSize() << std::endl;
      str << "Allocated elements:" << nonzeroElements. getSize() << std::endl;
      str << "Number of groups: " << groupOffsets. getSize() << std::endl;

      Index print_lines = lines;
      if( ! print_lines )
         print_lines = this->getSize();

      for( Index i = 0; i < this->numberOfGroups; i ++ )
      {
         if( i * groupSize > print_lines )
            return;
         str << std::endl << "Block number: " << i << std::endl;
         str << " Group size: " << this->getCurrentGroupSize( i ) << std::endl;
         str << " Group non-zeros: ";
         for( Index k = i * groupSize; k < ( i + 1 ) * groupSize && k < this->getSize(); k ++ )
            str << nonzeroElementsInRow. getElement( k ) << "  ";
         str << std::endl;
         str << " Group data: "
             << groupOffsets. getElement( i ) << " -- "
             << groupOffsets. getElement( i + 1 ) << std::endl;
         str << " Data:   ";
         for( Index k = groupOffsets. getElement( i );
              k < groupOffsets. getElement( i + 1 );
              k ++ )
            str << std::setprecision( 5 ) << std::setw( 8 )
                << nonzeroElements. getElement( k ) << " ";
         str << std::endl << "Columns: ";
         for( Index k = groupOffsets. getElement( i );
              k < groupOffsets. getElement( i + 1 );
              k ++ )
            str << std::setprecision( 5 ) << std::setw( 8 )
                << columns. getElement( k ) << " ";
         str << std::endl << "Offsets: ";
         for( Index k = groupOffsets. getElement( i );
              k < groupOffsets. getElement( i + 1 );
              k ++ )
            str << std::setprecision( 5 ) << std::setw( 8 )
                << k << " ";

      }
      str << std::endl;
      return;
   }
   if( format == "html" )
   {
      str << "<h1>Structure of tnlRgCSRMatrix</h1>" << std::endl;
      str << "<b>Matrix name:</b> " << name << "<p>" << std::endl;
      str << "<b>Matrix size:</b> " << this->getSize() << "<p>" << std::endl;
      str << "<b>Allocated elements:</b> " << nonzeroElements. getSize() << "<p>" << std::endl;
      str << "<b>Number of groups:</b> " << this->numberOfGroups << "<p>" << std::endl;
      str << "<table border=1>" << std::endl;
      str << "<tr> <td> <b> GroupId </b> </td> <td> <b> Size </b> </td> <td> <b> % of nonzeros </b> </td> </tr>" << std::endl;
      Index print_lines = lines;
      if( ! print_lines )
         print_lines = this->getSize();

      for( Index i = 0; i < this->numberOfGroups; i ++ )
      {
         if( i * groupSize > print_lines )
            return;
         double filling = ( double ) ( this->groupOffsets. getElement( i + 1 ) - this->groupOffsets. getElement( i ) ) /
                          ( double ) this->nonzeroElements. getSize();
         str << "<tr> <td> " << i << "</td> <td>" << this->getCurrentGroupSize( i ) << " </td> <td> " << 100.0 * filling << "% </td></tr>" << std::endl;
      }
      str << "</table>" << std::endl;
      str << std::endl;
   }
};

template< typename Real, typename Device, typename Index >
bool tnlRgCSRMatrix< Real, Device, Index > :: draw( std::ostream& str,
                                                    const String& format,
                                                    tnlCSRMatrix< Real, Device, Index >* csrMatrix,
                                                    int verbose )
{
   if( Device :: getDevice() == tnlCudaDevice )
   {
      std::cerr << "Drawing of matrices stored on the GPU is not supported yet." << std::endl;
      return false;
   }
   if( format == "gnuplot" )
      return tnlMatrix< Real, Device, Index > ::  draw( str, format, csrMatrix, verbose );
   if( format == "eps" )
   {
      const int elementSize = 10;
      this->writePostscriptHeader( str, elementSize );

      /****
       * Draw the groups
       */
      for( Index groupId = 0; groupId < numberOfGroups; groupId ++ )
      {
         const Index groupSize = getCurrentGroupSize( groupId );
         if( groupId % 2 == 0 )
            str << "0.9 0.9 0.9 setrgbcolor" << std::endl;
         else
            str << "0.8 0.8 0.8 setrgbcolor" << std::endl;
         str << "0 -" << groupSize * elementSize
             << " translate newpath 0 0 " << this->getSize() * elementSize
             << " " << groupSize * elementSize << " rectfill" << std::endl;
      }
      /****
       * Restore black color and the origin of the coordinates
       */
      str << "0 0 0 setrgbcolor" << std::endl;
      str << "0 " << this->getSize() * elementSize << " translate" << std::endl;

      if( csrMatrix )
         csrMatrix -> writePostscriptBody( str, elementSize, verbose );
      else
         this->writePostscriptBody( str, elementSize, verbose );

      str << "showpage" << std::endl;
      str << "%%EOF" << std::endl;

      if( verbose )
        std::cout << std::endl;
      return true;
   }
   std::cerr << "The format " << format << " is not supported for matrix drawing." << std::endl;
}

template< typename Real, typename Device, typename Index >
Index tnlRgCSRMatrix< Real, Device, Index > :: getCurrentGroupSize( const Index groupId ) const
{
   Assert( groupId < numberOfGroups, std::cerr << " groupId = " << groupId << " numberOfGroups = " << numberOfGroups );
   /****
    * If we use adaptive group sizes they are stored explicitly.
    */
   if( this->useAdaptiveGroupSize )
      return this->adaptiveGroupSizes. getElement( groupId + 1 ) -
             this->adaptiveGroupSizes. getElement( groupId );
   /****
    * The last group may be smaller even if we have constant group size.
    */
   if( ( groupId + 1 ) * this->groupSize > this->getSize() )
      return this->getSize() % this->groupSize;
   /***
    * If it is not the last group, return the common group size.
    */
   return this->groupSize;
}

template< typename Real, typename Device, typename Index >
Index tnlRgCSRMatrix< Real, Device, Index > :: getGroupIndexFromRow( const Index row ) const
{
   Assert( row < this->getSize(), std::cerr << " row = " << row << " matrix size = " << this->getSize() );
   if( this->useAdaptiveGroupSize )
   {
      Index groupId = -1;
      while( this->adaptiveGroupSizes. getElement( groupId + 1 ) <= row )
         groupId ++;
      return groupId;
   }
   return row / groupSize;
}

template< typename Real, typename Device, typename Index >
Index tnlRgCSRMatrix< Real, Device, Index > :: getRowIndexInGroup( const Index row, const Index groupId ) const
{
   Assert( row < this->getSize(), std::cerr << " row = " << row << " matrix size = " << this->getSize() );
   Assert( groupId < numberOfGroups, std::cerr << " groupId = " << groupId << " numberOfGroups = " << numberOfGroups );
   if( this->useAdaptiveGroupSize )
      return row - adaptiveGroupSizes. getElement( groupId );
   return row % groupSize;

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
__global__ void tnlRgCSRMatrixVectorProductKernel( const Index gridNumber,
                                                   const Index maxGridSize,
                                                   Index size,
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
   const Index rowIndex = gridNumber * maxGridSize * blockDim. x + blockIdx. x * blockDim. x + threadIdx. x;
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
__global__ void tnlRgCSRMatrixAdpativeGroupSizeVectorProductKernel( const Index gridNumber,
                                                                    const Index maxGridSize,
                                                                    Index size,
                                                                    const Index* groupsToRowsMapping,
                                                                    const Real* nonzeroElements,
                                                                    const Index* columns,
                                                                    const Index* groupOffsets,
                                                                    const Index* nonzerosInRow,
                                                                    const Real* vec_x,
                                                                    Real* vec_b )
{
   extern __shared__ int sdata[];
   Real* partialSums = reinterpret_cast< Real* >( sdata );
   /****
    * Now the group size is variable and one group is processed
    * by one block. We assume that the groupSize divides the blockDim. x
    */

   const Index blockIndex = blockIdx. x + gridNumber * maxGridSize;
   const Index firstRowInGroup = groupsToRowsMapping[ blockIndex ];
   const Index groupSize = groupsToRowsMapping[ blockIndex + 1 ] - firstRowInGroup;

   /****
    * The last group in the matrix is usually smaller and it does
    * not divide the BlockDim. x. In this case we leave the number
    * of threads per row 1.
    */
   Index threadsPerRow( 1 ), threadIndexInRow( 0 ), rowInGroup( threadIdx. x ), activeThreads( groupSize );
   if( blockDim. x % groupSize == 0 )
   {
      threadsPerRow = blockDim. x / groupSize;
      threadIndexInRow = threadIdx. x / groupSize;
      rowInGroup = threadIdx. x % groupSize;
      activeThreads = blockDim. x;
   }
   const Index rowInMatrix = firstRowInGroup + rowInGroup;

   /****
    * We need to call __syncthreads() later so we cannot do:
    * if( rowInMatrix >= size ) return;
    */
   Index nonzeros( 0 );
   if( rowInMatrix < size )
      nonzeros = nonzerosInRow[ rowInMatrix ];
   Index pos = groupOffsets[ blockIndex ] + rowInGroup + threadIndexInRow * groupSize;

   /****
    * Compute the partial sums
    */
   partialSums[ threadIdx. x ] = 0.0;
   for( Index i = threadIndexInRow; i < nonzeros; i += threadsPerRow )
   {
      const Index column = columns[ pos ];
      //if( column == -1 )
      //   printf( "* rowInMatrix = %d blockIdx. x = %d threadIdx. x = %d threadIndexInRow = %d i = %d \n",
      //          rowInMatrix, blockIndex, threadIdx. x, threadIndexInRow, i );
      if( column != -1 )
         partialSums[ threadIdx. x ] += nonzeroElements[ pos ] * vec_x[ column ];
      //if( rowInMatrix == 0 )
      //   printf( "partialSums[ %d ] = %f \n", threadIdx. x, partialSums[ threadIdx. x ] );
      pos += activeThreads;
   }
   __syncthreads();

   /****
    * There will be no other __syncthreads() so we may quit inactive threads.
    */
   if( rowInMatrix >= size )
      return;

   /****
    * Sum up the partial sums
    */
   /*Real pSum( 0.0 );
   if( threadIndexInRow == 0 )
   {
       pSum = partialSums[ threadIdx. x ];
      for( Index i = 1; i < threadsPerRow; i ++  )
      {
         pSum += partialSums[ threadIdx. x + i * groupSize ];
         //if( rowInMatrix == 0 )
         //   printf( "partialSums[ %d ] = %f i = %d \n", threadIdx. x, partialSums[ threadIdx. x ], i );

      }
   }*/



   if( threadIndexInRow + 1 < threadsPerRow )
      partialSums[ threadIdx. x ] += partialSums[ threadIdx. x + 1 * groupSize ];

   if( threadIndexInRow + 2 < threadsPerRow )
      partialSums[ threadIdx. x ] += partialSums[ threadIdx. x + 2 * groupSize ];

   if( threadIndexInRow + 4 < threadsPerRow )
      partialSums[ threadIdx. x ] += partialSums[ threadIdx. x + 4 * groupSize ];

   if( threadIndexInRow + 8 < threadsPerRow )
      partialSums[ threadIdx. x ] += partialSums[ threadIdx. x + 8 * groupSize ];

   if( threadIndexInRow + 16 < threadsPerRow )
      partialSums[ threadIdx. x ] += partialSums[ threadIdx. x + 16 * groupSize ];

   if( threadIndexInRow + 32 < threadsPerRow )
      partialSums[ threadIdx. x ] += partialSums[ threadIdx. x + 32 * groupSize ];

   if( threadIndexInRow + 64 < threadsPerRow )
      partialSums[ threadIdx. x ] += partialSums[ threadIdx. x + 64 * groupSize ];

   if( threadIndexInRow + 128 < threadsPerRow )
      partialSums[ threadIdx. x ] += partialSums[ threadIdx. x + 128 * groupSize ];

   if( threadIndexInRow + 256 < threadsPerRow )
      partialSums[ threadIdx. x ] += partialSums[ threadIdx. x + 256  * groupSize];

   if( threadIndexInRow + 512 < threadsPerRow )
      partialSums[ threadIdx. x ] += partialSums[ threadIdx. x + 512 * groupSize ];

   if( threadIndexInRow + 1024 < threadsPerRow )
      partialSums[ threadIdx. x ] += partialSums[ threadIdx. x + 1024 * groupSize ];

   if( threadIndexInRow == 0 )
   {
      //if( pSum != partialSums[ threadIdx. x ] )
      //   printf( "!!! pSum = %f partialSum = %f threadsPerRow = %d groupSize = %d rowInMatrix = %d\n", pSum, partialSums[ threadIdx. x ], threadsPerRow, groupSize, rowInMatrix );
      //printf( "partialSums[ %d ] =  %f rowInMatrix = %d \n ", threadIdx. x, partialSums[ threadIdx. x ], rowInMatrix );
      vec_b[ rowInMatrix ] = partialSums[ threadIdx. x ];
   }
}



#endif // ifdef HAVE_CUDA


#endif /* TNLRgCSRMATRIX_H_ */
