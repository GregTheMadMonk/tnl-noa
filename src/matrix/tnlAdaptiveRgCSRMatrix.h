/***************************************************************************
                          tnlAdaptiveRgCSRMatrix.h  -  description
                             -------------------
    begin                : Mar 19, 2011
    copyright            : (C) 2011 by Martin Heller
    email                : hellemar@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/


#ifndef TNLRgCSRMATRIX_H_
#define TNLRgCSRMATRIX_H_

#include <iostream>
#include <iomanip>
#include <core/tnlLongVectorHost.h>
#include <core/tnlAssert.h>
#include <core/mfuncs.h>
#include <matrix/tnlCSRMatrix.h>
#include <debug/tnlDebug.h>

using namespace std;

struct tnlARGCSRGroupProperties
{
   int numRows;
   //int numUsedThreads;
   int numRounds;
   int idxFirstRow;
   int idxFirstValue;
};

inline tnlString GetParameterType( const tnlARGCSRGroupProperties& a )
{
   return tnlString( "tnlARGCSRGroupProperties" );
}

//! Matrix storing the non-zero elements in the Row-grouped CSR (Compressed Sparse Row) format
/*!
 */
template< typename Real, tnlDevice Device = tnlHost, typename Index = int >
class tnlAdaptiveRgCSRMatrix : public tnlMatrix< Real, Device, Index >
{
   public:
   //! Basic constructor
   tnlAdaptiveRgCSRMatrix( const tnlString& name, 
                           Index _groupSize = 128,
			                  Index _groupSizeStep = 16,
			                  Index _targetNonzeroesPerGroup = 2048 );

   const tnlString& getMatrixClass() const;

   tnlString getType() const;

   Index getMaxGroupSize() const;

   Index getCUDABlockSize() const;

   //! This can only be a multiple of the groupSize
   void setCUDABlockSize( Index blockSize );

   //! Sets the number of row and columns.
   bool setSize( Index new_size );

   //! Allocate memory for the nonzero elements.
   bool setNonzeroElements( Index elements );

   Index getNonzeroElements() const;

   Index getArtificialZeroElements() const;

   Real getElement( Index row, Index column ) const;

   bool setElement( Index row,
                    Index colum,
                    const Real& value )
   { abort(); };

   bool addToElement( Index row,
                      Index column,
                      const Real& value )
   { abort(); };

   Real rowProduct( const Index row,
                    const tnlLongVector< Real, Device, Index >& vec ) const
   { abort(); };

   void vectorProduct( const tnlLongVector< Real, Device, Index >& vec,
                       tnlLongVector< Real, Device, Index >& result ) const;

   bool copyFrom( const tnlCSRMatrix< Real, tnlHost,
                  Index >& csr_matrix,
                  const Index cudaBlockSize = 256 );

   template< tnlDevice Device2 >
   bool copyFrom( const tnlAdaptiveRgCSRMatrix< Real, Device2, Index >& rgCSRMatrix );

   Real getRowL1Norm( Index row ) const
   { abort(); };

   void multiplyRow( Index row, const Real& value )
   { abort(); };

   //! Prints out the matrix structure
   void printOut( ostream& str,
		          const Index lines = 0 ) const;

   protected:

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

   tnlLongVector< Real, Device, Index > nonzeroElements;

   tnlLongVector< Index, Device, Index > columns;

   tnlLongVector< Index, Device, Index > threads;

   tnlLongVector< tnlARGCSRGroupProperties, Device, Index > groupInfo;

   tnlLongVector< Index, tnlHost, Index > usedThreadsInGroup;

   tnlLongVector< Index, Device, Index > rowToGroupMapping;

   Index maxGroupSize, groupSizeStep;

   Index targetNonzeroesPerGroup;

   Index numberOfGroups;

   Index cudaBlockSize;

   Index artificial_zeros;

   //! The last non-zero element is at the position last_non_zero_element - 1
   Index last_nonzero_element;

   friend class tnlAdaptiveRgCSRMatrix< Real, tnlHost, Index >;
   friend class tnlAdaptiveRgCSRMatrix< Real, tnlCuda, Index >;
};

#ifdef HAVE_CUDA

template< class Real, typename Index, bool useCache >
__global__ void AdaptiveRgCSRMatrixVectorProductKernel( Real* target, 
    				                           		        const Real* vect,
							                                   const Real* matrxValues,
							                                   const Index* matrxColumni,
							                                   const Index* _groupInfo,
							                                   const Index* _threadsInfo,
							                                   const Index numBlocks );

#endif


template< typename Real, tnlDevice Device, typename Index >
tnlAdaptiveRgCSRMatrix< Real, Device, Index > :: tnlAdaptiveRgCSRMatrix( const tnlString& name,
	                                                                      Index _maxGroupSize,
		                                             						    Index _groupSizeStep,
									                                              Index _targetNonzeroesPerGroup )
: tnlMatrix< Real, Device, Index >( name ),
  nonzeroElements( name + " : nonzeroElements" ),
  columns( name + " : columns" ),
  threads( name + " : threads" ),
  groupInfo( name + ": groupInfo" ),
  usedThreadsInGroup( name + " : usedThreadsInGroup" ),
  rowToGroupMapping( name + " : rowToGroupMapping" ),
  maxGroupSize( _maxGroupSize ),
  groupSizeStep(_groupSizeStep),
  targetNonzeroesPerGroup(_targetNonzeroesPerGroup),
  numberOfGroups( 0 ),
  cudaBlockSize( 0 ),
  artificial_zeros( 0 ),
  last_nonzero_element( 0 )
{
	tnlAssert( maxGroupSize > 0, );
};

template< typename Real, tnlDevice Device, typename Index >
const tnlString& tnlAdaptiveRgCSRMatrix< Real, Device, Index > :: getMatrixClass() const
{
   return tnlMatrixClass :: main;
};

template< typename Real, tnlDevice Device, typename Index >
tnlString tnlAdaptiveRgCSRMatrix< Real, Device, Index > :: getType() const
{
   return tnlString( "tnlAdaptiveRgCSRMatrix< ") +
          tnlString( GetParameterType( Real( 0.0 ) ) ) +
          tnlString( ", " ) +
          getDeviceType( Device ) +
          tnlString( ", " ) +
          GetParameterType( Index( 0 ) ) +
          tnlString( " >" );
};

template< typename Real, tnlDevice Device, typename Index >
Index tnlAdaptiveRgCSRMatrix< Real, Device, Index > :: getMaxGroupSize() const
{
   return maxGroupSize;
}

template< typename Real, tnlDevice Device, typename Index >
Index tnlAdaptiveRgCSRMatrix< Real, Device, Index > :: getCUDABlockSize() const
{
   return cudaBlockSize;
}

template< typename Real, tnlDevice Device, typename Index >
bool tnlAdaptiveRgCSRMatrix< Real, Device, Index > :: setSize( Index new_size )
{
   this -> size = new_size;
   if( ! groupInfo. setSize( this -> getSize()) ||
       ! usedThreadsInGroup. setSize( this -> getSize() ) ||
       ! threads. setSize( this -> getSize() ) ||
       ! rowToGroupMapping. setSize( this -> getSize() ) )
      return false;
   threads. setValue( 0 );
   usedThreadsInGroup. setValue( 0 );
   rowToGroupMapping. setValue( 0 );
   last_nonzero_element = 0;
   return true;
};

template< typename Real, tnlDevice Device, typename Index >
bool tnlAdaptiveRgCSRMatrix< Real, Device, Index > :: setNonzeroElements( Index elements )
{
   tnlAssert( elements != 0, );
   if( ! nonzeroElements. setSize( elements ) ||
       ! columns. setSize( elements ) )
      return false;
   nonzeroElements. setValue( 0.0 );
   columns. setValue( -1 );
   return true;
};

template< typename Real, tnlDevice Device, typename Index >
Index tnlAdaptiveRgCSRMatrix< Real, Device, Index > :: getNonzeroElements() const
{
   tnlAssert( nonzeroElements. getSize() > artificial_zeros, );
	return nonzeroElements. getSize() - artificial_zeros;
}

template< typename Real, tnlDevice Device, typename Index >
Index tnlAdaptiveRgCSRMatrix< Real, Device, Index > :: getArtificialZeroElements() const
{
	return artificial_zeros;
}

template< typename Real, tnlDevice Device, typename Index >
bool tnlAdaptiveRgCSRMatrix< Real, Device, Index > :: copyFrom( const tnlCSRMatrix< Real, tnlHost, Index >& mat,
                                                                const Index cudaBlockSize )
{
	dbgFunctionName( "tnlAdaptiveRgCSRMatrix< Real, tnlHost >", "copyFrom" );
	this -> cudaBlockSize = cudaBlockSize;
	if( ! this -> setSize( mat.getSize() ) )
		return false;
	
	Index nonzerosInGroup( 0 );
	Index groupBegin( 0 );
	Index groupEnd( 0 );
	Index rowsInGroup( 0 );
	Index groupId( 0 );

	Index numberOfStoredValues( 0 );
	Index threadsPerRow[ 128 ];
	
	/****
	 * This loop computes sizes of the groups and the number of threads per one row
	 */
	while( true )
	{
		/****
		 * First compute the group size such that the number of the non-zero elements in each group is
		 * approximately the same.
		 */
		groupEnd += 16;
		if( groupEnd > this -> getSize() )
		   groupEnd = this -> getSize();

		nonzerosInGroup = mat. row_offsets[ groupEnd ] - mat. row_offsets[ groupBegin ];
		rowsInGroup = groupEnd - groupBegin;

		if( nonzerosInGroup < targetNonzeroesPerGroup &&
		    groupEnd < this -> getSize() &&
		    rowsInGroup < maxGroupSize)
		   continue;

		dbgExpr( groupBegin );
		dbgExpr( groupEnd );
		dbgExpr( nonzerosInGroup );

		/****
		 * Now, compute the number of threads per each row.
		 * Each row get one thread by default.
		 * Then each row will get additional threads relatively to the
		 * number of the nonzero elements in the row.
		 */
		Index freeThreads = cudaBlockSize - rowsInGroup;
		Index usedThreads = 0;
		for( Index i = groupBegin; i < groupEnd; i++ )
		{
		   double nonzerosInRow = mat. getNonzeroElementsInRow( i );
		   double nonzerosInRowRatio = nonzerosInRow / ( double ) nonzerosInGroup;
		   usedThreads += threadsPerRow[ i - groupBegin ] = floor( freeThreads * nonzerosInRowRatio );
		}
		/****
		 * If there are some threads left distribute them to the rows from the group begining.
		 */
		Index threadsLeft = cudaBlockSize - usedThreads;
		dbgExpr( usedThreads );
		dbgExpr( threadsLeft );
		for( Index i=0; i < threadsLeft; i++)
			threadsPerRow[ i ]++;

		/****
		 * Compute prefix-sum on threadsPerRow and store it in threads
		 */
		threads[ groupBegin ] = threadsPerRow[ 0 ];
		for( Index i = groupBegin + 1; i< groupEnd; i++ )
		{
			threads[ i ] = threads[ i - 1 ] + threadsPerRow[ i - groupBegin ];
			dbgExpr( threads[ i ] );
		}
		usedThreadsInGroup[ groupId ] = threads[ groupEnd - groupBegin - 1 ]; // ???????
		dbgExpr( usedThreadsInGroup[ groupId ] );

		/****
		 * Now, compute the number of rounds
		 */
		Index rounds( 0 ), roundsFinal( 0 );
		for( Index i = groupBegin; i < groupEnd; i++ )
		{
		   double nonzerosInRow = mat. getNonzeroElementsInRow( i );
		   rounds = ceil( nonzerosInRow / ( double ) threadsPerRow[ i - groupBegin ] );
		   roundsFinal = Max( rounds, roundsFinal );
		}
		dbgExpr( roundsFinal );
		groupInfo[ groupId ]. numRows = rowsInGroup;
      groupInfo[ groupId ]. idxFirstRow = groupBegin;
      groupInfo[ groupId ]. idxFirstValue = numberOfStoredValues;
		groupInfo[ groupId ]. numRounds = roundsFinal;

		dbgExpr( groupInfo[ groupId ]. numRows );
		dbgExpr( groupInfo[ groupId ]. idxFirstRow );
		dbgExpr( groupInfo[ groupId ]. idxFirstValue );
		dbgExpr( groupInfo[ groupId ]. numRounds );

		for( Index i = groupBegin; i < groupEnd; i ++ )
		   rowToGroupMapping[ i ] = groupId;

		groupId++;
		numberOfStoredValues += cudaBlockSize * roundsFinal;
		groupBegin = groupEnd;

		if( groupBegin == this -> getSize() )
		{
			numberOfGroups = groupId;
			break;
		}
	}

	/****
	 * Allocate the non-zero elements (they contains some artificial zeros.)
	 */
	dbgCout( "Allocating " << numberOfStoredValues << " elements.");
	if( ! setNonzeroElements( numberOfStoredValues ) )
		return false;
	artificial_zeros = numberOfStoredValues - mat. getNonzeroElements();

	last_nonzero_element = numberOfStoredValues;

	dbgCout( "Inserting data " );
	if( Device == tnlHost )
	{
	   Index counters[ 128 ];
		Index NZperRow[ 128 ];
		Index index, baseRow;
		for( Index i = 0; i < numberOfGroups; i++ )
		{
			baseRow = groupInfo[ i ]. idxFirstRow;
			index = groupInfo[ i ]. idxFirstValue;
			dbgExpr( baseRow );
			dbgExpr( index );
			/****
			 * First compute number of threads for each row.
			 */
			for( Index j = 0; j < groupInfo[ i ]. numRows; j++ )
			{
			   NZperRow[ j ] = mat. getNonzeroElementsInRow( baseRow + j );
			   if( j > 0 )
			      threadsPerRow[ j ] = threads[ baseRow + j ] - threads[ baseRow + j - 1 ];
			   else
			      threadsPerRow[ j ] = threads[ baseRow ];
				counters[ j ] = 0;
			}
			/****
			 * Now do the insertion
			 */
			for( Index k = 0; k < groupInfo[ i ]. numRounds; k ++ )
				for( Index j = 0; j < groupInfo[ i ]. numRows; j ++ )
					for( Index l = 0; l < threadsPerRow[ j ]; l ++ )
					{
						if( counters[ j ] < NZperRow[ j ] )
						{
						   Index pos = mat. row_offsets[ baseRow + j ] + counters[ j ];
						   dbgCout( "Inserting data from CSR format at position " << pos << " to AdaptiveRgCSR at " << index );
							nonzeroElements[ index ] = mat. nonzero_elements[ pos ];
							columns[ index ] = mat.columns[ pos ];
						}
						else
						{
                     dbgCout( "Inserting artificial zero to AdaptiveRgCSR at " << index );
							columns[ index ] = -1;
							nonzeroElements[ index ] = 0.0;
						}
						counters[ j ] ++;
						index ++;
					}
		}
	}
	if( Device == tnlCuda )
	{
		tnlAssert( false,
			cerr << "Conversion from tnlCSRMatrix on the host to the tnlAdaptiveRgCSRMatrix on the CUDA device is not implemented yet."; );
		//TODO: implement this
	}
	return true;
}


template< typename Real, tnlDevice Device, typename Index >
   template< tnlDevice Device2 >
bool tnlAdaptiveRgCSRMatrix< Real, Device, Index > :: copyFrom( const tnlAdaptiveRgCSRMatrix< Real, Device2, Index >& adaptiveRgCSRMatrix )
{
   dbgFunctionName( "tnlAdaptiveRgCSRMatrix< Real, Device, Index >", "copyFrom" );
   maxGroupSize = adaptiveRgCSRMatrix.maxGroupSize;
   groupSizeStep = adaptiveRgCSRMatrix.groupSizeStep;
   targetNonzeroesPerGroup = adaptiveRgCSRMatrix.targetNonzeroesPerGroup;
   cudaBlockSize = adaptiveRgCSRMatrix.cudaBlockSize;
   last_nonzero_element = adaptiveRgCSRMatrix.last_nonzero_element;
   numberOfGroups = adaptiveRgCSRMatrix.numberOfGroups;
  

   if( ! this -> setSize( adaptiveRgCSRMatrix. getSize() ) )
      return false;   

   /****
    * Allocate the non-zero elements (they contains some artificial zeros.)
    */
   Index total_elements = adaptiveRgCSRMatrix. getNonzeroElements() + 
                          adaptiveRgCSRMatrix. getArtificialZeroElements() ;
   dbgCout( "Allocating " << total_elements << " elements.");
   if( ! setNonzeroElements( total_elements ) )
      return false;
   artificial_zeros = total_elements - adaptiveRgCSRMatrix. getNonzeroElements();

   nonzeroElements = adaptiveRgCSRMatrix.nonzeroElements;
   columns = adaptiveRgCSRMatrix.columns;
   groupInfo = adaptiveRgCSRMatrix.groupInfo;
   threads = adaptiveRgCSRMatrix.threads;

   return true;
};

template< typename Real, tnlDevice Device, typename Index >
Real tnlAdaptiveRgCSRMatrix< Real, Device, Index > :: getElement( Index row,
                                                                  Index column ) const
{
   tnlAssert( 0 <= row && row < this -> getSize(),
            cerr << "The row is outside the matrix." );
   if( Device == tnlHost )
   {
      Index groupId = rowToGroupMapping[ row ];
      Index groupRow = row - groupInfo[ groupId ]. idxFirstRow;
      Index groupOffset = groupInfo[ groupId ]. idxFirstValue;
      Index firstThread = 0;
      if( row > 0 )
         firstThread = threads[ row - 1 ];
      const Index lastThread = threads[ row ];
      const Index chunkSize = groupInfo[ groupId ]. numRounds;
      for( Index thread = firstThread; thread < lastThread; thread ++ )
         for( Index i = 0; i < chunkSize; i ++ )
         {
            Index pos = thread * chunkSize + i + groupInfo[ groupId ]. idxFirstValue;
            if( columns[ pos ] == column )
               return nonzeroElements[ pos ];
         }
      return 0.0;
   }
   if( Device == tnlCuda )
   {
      tnlAssert( false,
                cerr << "tnlRgCSRMatrix< Real, tnlCuda, Index > ::getElement is not implemented yet." );
      //TODO: implement this

   }
}

template< typename Real, tnlDevice Device, typename Index >
void tnlAdaptiveRgCSRMatrix< Real, Device, Index > :: vectorProduct( const tnlLongVector< Real, Device, Index >& vec,
                                                                     tnlLongVector< Real, Device, Index >& result ) const
{
   dbgFunctionName( "tnlAdaptiveRgCSRMatrix< Real, tnlHost >", "vectorProduct" )
   tnlAssert( vec. getSize() == this -> getSize(),
              cerr << "The matrix and vector for a multiplication have different sizes. "
                   << "The matrix size is " << this -> getSize() << "."
                   << "The vector size is " << vec. getSize() << endl; );
   tnlAssert( result. getSize() == this -> getSize(),
              cerr << "The matrix and result vector of a multiplication have different sizes. "
                   << "The matrix size is " << this -> getSize() << "."
                   << "The vector size is " << vec. getSize() << endl; );


   const Index TB_SIZE = 256;
   const Index MAX_ROWS = 128;
   if( Device == tnlHost )
   {
      Index idx[TB_SIZE];
      Real psum[TB_SIZE];        //partial sums for each thread
      Index limits[MAX_ROWS + 1];  //indices of first threads for each row + index of first unused thread
      Real results[MAX_ROWS];

      /****
       * Go over all groups ...
       */
      for( Index group = 0; group < this -> numberOfGroups; group ++ )
      {
         /****
          * In each group compute partial sums of each thread
          */
         for( Index thread = 0;
              thread < this -> usedThreadsInGroup[ group ];
              thread ++ )
         {
            idx[ thread ] = this -> groupInfo[ group ]. idxFirstValue + thread;
            psum[thread] = 0;

            for( Index j = 0;
                 j < this -> groupInfo[ group ]. numRounds;
                 j ++ )
            {
               psum[ thread ] += this -> nonzeroElements[ idx[ thread ] ] * vec[ this -> columns[ idx[ thread ] ] ];
               idx[ j ] += TB_SIZE;
            }
         }

         /****
          * Compute local copy of thread indexes mapped to given row of the group.
          * (this is only to simulate copying data to the fast shared memory on GPU)
          */
         for( Index thread = 0;
              thread < this -> groupInfo[ group ]. numRows;
              thread ++ )
            limits[ thread ] = this -> threads[ this -> groupInfo[ group ]. idxFirstRow + thread ];
         /****
          * For convenience, add the index of first unused row.
          */
         limits[ this -> groupInfo[ group ]. numRows ] = this -> usedThreadsInGroup[ group ];

         //reduction of partial sums and writing to the output
         for( Index thread = 0;
              thread < this -> groupInfo[ group ]. numRows;
              thread ++)         //for threads corresponding to rows in group
         {
            results[ thread ] = 0;
            for( Index j = limits[ thread ];
                 j < limits[ thread + 1 ];
                 j++ )              //sum up partial sums belonging to that row
               results[ thread ] += psum[ j ];
            result[ this -> groupInfo[ group ]. idxFirstRow + thread ] = results[ thread ];
         }
      }
   }
   if( Device == tnlCuda )
   {
#ifdef HAVE_CUDA
   Index blockSize = this -> getCUDABlockSize();
   const Index size = this -> getSize();

   Index desGridSize;
	desGridSize = this->numberOfGroups;
	desGridSize = (desGridSize < 4096) ? desGridSize : 4096;

   cudaThreadSynchronize();
   int gridSize = (int) desGridSize;
   dim3 gridDim( gridSize ), blockDim( blockSize );

   /*AdaptiveRgCSRMatrixVectorProductKernel< Real, Index, false ><<< gridDim, blockDim >>>( result. getVector(),
											  vec. getVector(),
                                                                                          nonzeroElements. getVector(),
                                                                                          columns. getVector(),
                                                                                          groupInfo. getVector(),
                                                                                          threadsPerRow. getVector(),
	                                        					  numberOfGroups );*/
    cudaThreadSynchronize();
    CHECK_CUDA_ERROR;
#else
   cerr << "CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif
   }

}

template< typename Real, tnlDevice Device, typename Index >
void tnlAdaptiveRgCSRMatrix< Real, Device, Index > :: printOut( ostream& str,
		                                                          const Index lines ) const
{
   str << "Structure of tnlAdaptiveRgCSRMatrix" << endl;
   str << "Matrix name:" << this -> getName() << endl;
   str << "Matrix size:" << this -> getSize() << endl;
   str << "Allocated elements:" << nonzeroElements. getSize() << endl;
   str << "Number of groups: " << numberOfGroups << endl;

   Index print_lines = lines;
   if( ! print_lines )
	   print_lines = this -> getSize();

   for( Index groupId = 0; groupId < numberOfGroups; groupId ++ )
   {
      const Index firstRow = groupInfo[ groupId ]. idxFirstRow;
      const Index lastRow = firstRow + groupInfo[ groupId ]. numRows;
	   if( firstRow  > print_lines )
		   return;
	   str << endl << "Group number: " << groupId << endl;
	   str << " Rows: " << firstRow << " -- " << lastRow << endl;
	   str << " Chunk size: " << groupInfo[ groupId ]. numRounds << endl;
	   str << " Threads per row: ";
	   for( Index row = firstRow; row < lastRow; row ++ )
		   str << threads. getElement( row ) << "  ";
	   str << endl;
      str << " Group offset: " << groupInfo[ groupId ]. idxFirstValue <<  endl;
      for( Index row = firstRow; row < lastRow; row ++ )
      {
         str << " Data for row number " << row << ": ";
         Index firstThread = 0;
         if( row > 0 )
            firstThread = threads[ row - 1 ];
         Index lastThread = threads[ row ];
         const Index chunkSize = groupInfo[ groupId ]. numRounds;
         for( Index thread = firstThread; thread < lastThread; thread ++ )
            for( Index i = 0; i < chunkSize; i ++ )
            {
               Index pos = thread * chunkSize + i + groupInfo[ groupId ]. idxFirstValue;
               str << nonzeroElements[ pos ] << " ( " << columns[ pos ] << " ) ";
            }
         str << endl;
      }
   }
   str << endl;
};

#ifdef HAVE_CUDA

template< class Real, typename Index, bool useCache >  // useCache unnecessary, we read x from global memory
__global__ void AdaptiveRgCSRMatrixVectorProductKernel( Real* target,
                                                        const Real* vect,
                                                        const Real* nonzeroElements,
                                                        const Index* columns,
                                                        const Index* _groupInfo,
                                                        const Index* _threadsInfo, 
                                                        const Index numBlocks )
{

	__shared__ Real partialSums[ 256 ];
	__shared__ Index info[ 4 ];			// first row index, number of rows assigned to the block, number of "rounds", first value and col index
	__shared__ Index threadsInfo[ 129 ];

	Index idx, begin, end, column;
	Real vectVal, sum;

	for( Index bId = blockIdx.x; bId < numBlocks; bId += gridDim.x)
	{
	   /****
	    * Read the group info from the global memory
	    */
		if( threadIdx.x < 4 )
			info[ threadIdx.x ] = _groupInfo[ 4 * bId + threadIdx.x ];
		__syncthreads();


		/****
		 * Read mapping of threads to rows.
		 * It says IDs of threads that will work on each row.
		 */
		if( threadIdx. x < info[ 1 ] )
		   threadsInfo[ threadIdx.x ] = _threadsInfo[ info[ 0 ] + threadIdx.x ];
		if( threadIdx. x == info[ 1 ] )
			threadsInfo[ threadIdx. x ] = blockDim. x;

		/****
		 * Each thread now computes partial sum in its chunk
		 */
		sum = 0;
		for( Index i = 0; i < info[ 2 ]; i ++ )
		{
			idx = threadIdx. x + i * blockDim. x + info[ 3 ];
			column = columns[ idx ];
			if( column != -1 )
			{
				vectVal = vect[ column ];
				sum += nonzeroElements[ idx ] * vectVal;
			}
		}
		partialSums[ threadIdx. x ] = sum;
		__syncthreads();

		/****
		 * Now sum the partial sums in each row
		 */
		if( threadIdx. x < info[ 1 ] )
		{
			sum = 0;
			begin = threadsInfo[ threadIdx.x ];
			end = threadsInfo[ threadIdx.x + 1 ];
			for( Index i = begin; i < end; i++ )
				sum += partialSums[ i ];

			target[ info[ 0 ] + threadIdx. x ] = sum;
		}
	}
}

#endif // ifdef HAVE_CUDA


#endif /* TNLRgCSRMATRIX_H_ */
