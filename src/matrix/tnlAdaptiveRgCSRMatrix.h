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

   Real getElement( Index row, Index column ) const
   { abort(); };

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

   bool copyFrom( const tnlCSRMatrix< Real, tnlHost, Index >& csr_matrix );

   template< tnlDevice Device2 >
   bool copyFrom( const tnlAdaptiveRgCSRMatrix< Real, Device2, Index >& rgCSRMatrix );

   Real getRowL1Norm( Index row ) const
   { abort(); };

   void multiplyRow( Index row, const Real& value )
   { abort(); };

   //! Prints out the matrix structure
   void printOut( ostream& str,
		          const Index lines = 0 ) const {};

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

   tnlLongVector< Real, Device, Index > nonzero_elements;

   tnlLongVector< Index, Device, Index > columns;

   tnlLongVector< Index, Device, Index > block_info; // size 4*number of groups; index of first row in group, nuber of rows in group, 
																				//	number of rounds, index of first nz element in group

   tnlLongVector< Index, Device, Index > threads_per_row;

   Index maxGroupSize, groupSizeStep;
   Index targetNonzeroesPerGroup;

   Index number_of_groups;

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
							const Index* _blockInfo, 
							const Index* _threadsInfo, 
							const Index numBlocks);

#endif


template< typename Real, tnlDevice Device, typename Index >
tnlAdaptiveRgCSRMatrix< Real, Device, Index > :: tnlAdaptiveRgCSRMatrix( const tnlString& name,
	                                                                 Index _maxGroupSize, 
								         Index _groupSizeStep, 
									 Index _targetNonzeroesPerGroup )
: tnlMatrix< Real, Device, Index >( name ),
  nonzero_elements( "nonzero-elements" ),
  columns( "columns" ),
  block_info( "block-info" ),
  threads_per_row( "threads-per-row" ),
  maxGroupSize( _maxGroupSize ),
  groupSizeStep(_groupSizeStep),
  targetNonzeroesPerGroup(_targetNonzeroesPerGroup),
  number_of_groups( 0 ),
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
void tnlAdaptiveRgCSRMatrix< Real, Device, Index > :: setCUDABlockSize( Index blockSize )
{
   tnlAssert( blockSize >= maxGroupSize, );
   cudaBlockSize = blockSize;
}

template< typename Real, tnlDevice Device, typename Index >
bool tnlAdaptiveRgCSRMatrix< Real, Device, Index > :: setSize( Index new_size )
{
   this -> size = new_size;
   if( ! block_info.setSize(this->size) ||  ! threads_per_row.setSize(this->size) )
      return false;
   block_info.setValue( 0 );
   threads_per_row.setValue( 0 );
   last_nonzero_element = 0;
   return true;
};

template< typename Real, tnlDevice Device, typename Index >
bool tnlAdaptiveRgCSRMatrix< Real, Device, Index > :: setNonzeroElements( Index elements )
{
   tnlAssert( elements !=0, );
   if( ! nonzero_elements.setSize(elements) ||  ! columns.setSize(elements) )
      return false;
   nonzero_elements.setValue( 0.0 );
   columns.setValue( -1 );
   return true;
};

template< typename Real, tnlDevice Device, typename Index >
Index tnlAdaptiveRgCSRMatrix< Real, Device, Index > :: getNonzeroElements() const
{
   tnlAssert( nonzero_elements. getSize() > artificial_zeros, );
	return nonzero_elements. getSize() - artificial_zeros;
}

template< typename Real, tnlDevice Device, typename Index >
Index tnlAdaptiveRgCSRMatrix< Real, Device, Index > :: getArtificialZeroElements() const
{
	return artificial_zeros;
}

template< typename Real, tnlDevice Device, typename Index >
bool tnlAdaptiveRgCSRMatrix< Real, Device, Index > :: copyFrom( const tnlCSRMatrix< Real, tnlHost, Index >& mat )
{
	dbgFunctionName( "tnlAdaptiveRgCSRMatrix< Real, tnlHost >", "copyFrom" );
	tnlAssert( cudaBlockSize != 0, );
	if( ! this -> setSize( mat.getSize() ) )
		return false;
	
	uint blkNZ = 0, blkBegin = 0, blkEnd = 0, rowsInBlk = 0;
	uint blkIdx = 0;

	uint numStoredValues = 0;
	uint threadsPerRow[128];
	
	/****
	 * This loop computes sizes of the groups  the number of threads per one row
	 */
	while(true) 
	{
		/****
		 * First compute the group size such that the number of the non-zero elemnts in each group is
		 * approximately the same.
		 */
		blkEnd += 16;
		if(blkEnd > this->size) {
			blkEnd = this->size;
		}
		blkNZ = mat.row_offsets[blkEnd] - mat.row_offsets[blkBegin];
		rowsInBlk = blkEnd - blkBegin;

		if(blkNZ < targetNonzeroesPerGroup && blkEnd < this->size && rowsInBlk < maxGroupSize) {
			continue;
		}

		/****
		 * Now, compute the number of threads per each row
		 */
		block_info[4*blkIdx] = blkBegin;     // group begining
		block_info[4*blkIdx+1] = rowsInBlk;  // number of the rows in the group

		uint freeThreads = cudaBlockSize - rowsInBlk;  // T -r 
		uint usedThreads = 0;
		for(uint i=blkBegin; i<blkEnd; i++) {
			usedThreads += (threadsPerRow[i-blkBegin] = floor(freeThreads * ((double) mat.row_offsets[i+1] - mat.row_offsets[i]) / blkNZ) + 1);
		}
		uint threadsLeft = cudaBlockSize - usedThreads;
		for(uint i=0; i<threadsLeft; i++) {
			threadsPerRow[i]++;
		}
		threads_per_row[blkBegin]=0;
		for(uint i=blkBegin+1; i<blkEnd; i++) {
			threads_per_row[i] = threads_per_row[i-1] + threadsPerRow[i-blkBegin-1];
		}			

		/****
		 * Now, compute the number of rounds
		 */
		uint rounds = 0, roundsFinal = 0;
		for(uint i=blkBegin; i<blkEnd; i++) {
			rounds = ceil(((double) mat.row_offsets[i+1] - mat.row_offsets[i]) / threadsPerRow[i-blkBegin]);
			roundsFinal = (rounds>roundsFinal) ? rounds : roundsFinal;
		}
		block_info[4*blkIdx+2] = roundsFinal;
		block_info[4*blkIdx+3] = numStoredValues;
		blkIdx++;
		numStoredValues += cudaBlockSize * roundsFinal;
		blkBegin = blkEnd;

		if(blkBegin == this->size) {
			number_of_groups = blkIdx;
			break;
		}
	}

	/****
	 * Allocate the non-zero elements (they contains some artificial zeros.)
	 */
	dbgCout( "Allocating " << numStoredValues << " elements.");
	if( ! setNonzeroElements( numStoredValues ) )
		return false;
	artificial_zeros = numStoredValues - mat.getNonzeroElements();

	last_nonzero_element = numStoredValues;

	dbgCout( "Inserting data " );
	if( Device == tnlHost )
	{
		uint counters[128];
		uint NZperRow[128];
		uint index, baseRow;
		for(uint i=0; i<number_of_groups; i++) {
			baseRow = block_info[4*i];
			index = block_info[4*i+3];
			for(uint j=0; j<block_info[4*i+1]; j++) {
				NZperRow[j] = mat.row_offsets[block_info[4*i]+j+1] - mat.row_offsets[block_info[4*i]+j];
				if(j<block_info[4*i+1]-1) {
					threadsPerRow[j] = threads_per_row[block_info[4*i]+j+1] - threads_per_row[block_info[4*i]+j];
				}
				else threadsPerRow[j] = cudaBlockSize - threads_per_row[block_info[4*i]+j];
				counters[j] = 0;
			}
			for(uint k=0; k<block_info[4*i+2]; k++) {
				for(uint j=0; j<block_info[4*i+1]; j++) {
					for(uint l=0; l<threadsPerRow[j]; l++) {
						if(counters[j]<NZperRow[j]) {
							nonzero_elements[index] = mat.nonzero_elements[ mat.row_offsets[baseRow+j]+counters[j] ];
							columns[index] = mat.columns[ mat.row_offsets[baseRow+j]+counters[j] ];
						}
						else {
							columns[index] = -1;
							nonzero_elements[index] = 0.0;
						}
						counters[j]++;
						index++;
					}
				}				
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
   number_of_groups = adaptiveRgCSRMatrix.number_of_groups;
  

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

   nonzero_elements = adaptiveRgCSRMatrix.nonzero_elements;
   columns = adaptiveRgCSRMatrix.columns;
   block_info = adaptiveRgCSRMatrix.block_info;
   threads_per_row = adaptiveRgCSRMatrix.threads_per_row;

   return true;
};


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

   if( Device == tnlHost )
   {
      abort();
   }
   if( Device == tnlCuda )
   {
#ifdef HAVE_CUDA
   Index blockSize = this -> getCUDABlockSize();
   const Index size = this -> getSize();

   Index desGridSize;
	desGridSize = this->number_of_groups;
	desGridSize = (desGridSize < 16384) ? desGridSize : 16384;

   cudaThreadSynchronize();
   int gridSize = (int) desGridSize;
   dim3 gridDim( gridSize ), blockDim( blockSize );

   AdaptiveRgCSRMatrixVectorProductKernel< Real, Index, false ><<< gridDim, blockDim >>>( result. getVector(),
											  vec. getVector(),
                                                                                          nonzero_elements. getVector(),
                                                                                          columns. getVector(),
                                                                                          block_info. getVector(),
                                                                                          threads_per_row. getVector(),
											  number_of_groups );
    cudaThreadSynchronize();
    CHECK_CUDA_ERROR;
#else
   cerr << "CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif
   }

}

//template< typename Real, tnlDevice Device, typename Index >
//void tnlAdaptiveRgCSRMatrix< Real, Device, Index > :: printOut( ostream& str,
//		                                                 const Index lines ) const
//{
//   str << "Structure of tnlAdaptiveRgCSRMatrix" << endl;
//   str << "Matrix name:" << this -> getName() << endl;
//   str << "Matrix size:" << this -> getSize() << endl;
//   str << "Allocated elements:" << nonzero_elements. getSize() << endl;
//   str << "Matrix blocks: " << block_offsets. getSize() << endl;
//
//   Index print_lines = lines;
//   if( ! print_lines )
//	   print_lines = this -> getSize();
//
//   for( Index i = 0; i < this -> block_offsets. getSize() - 1; i ++ )
//   {
//	   if( i * groupSize > print_lines )
//		   return;
//	   str << endl << "Block number: " << i << endl;
//	   str << " Lines: " << i * groupSize << " -- " << ( i + 1 ) * groupSize << endl;
//	   str << " Lines non-zeros: ";
//	   for( Index k = i * groupSize; k < ( i + 1 ) * groupSize && k < this -> getSize(); k ++ )
//		   str << nonzeros_in_row. getElement( k ) << "  ";
//	   str << endl;
//	   str << " Block data: "
//	       << block_offsets. getElement( i ) << " -- "
//	       << block_offsets. getElement( i + 1 ) << endl;
//	   str << " Data:   ";
//	   for( Index k = block_offsets. getElement( i );
//	        k < block_offsets. getElement( i + 1 );
//	        k ++ )
//		   str << setprecision( 5 ) << setw( 8 )
//		       << nonzero_elements. getElement( k ) << " ";
//	   str << endl << "Columns: ";
//	   for( Index k = block_offsets. getElement( i );
//	        k < block_offsets. getElement( i + 1 );
//	        k ++ )
//		   str << setprecision( 5 ) << setw( 8 )
//		       << columns. getElement( k ) << " ";
//   }
//   str << endl;
//};

#ifdef HAVE_CUDA

template< class Real, typename Index, bool useCache >  // useCache unnecessary, we read x from global memory
__global__ void AdaptiveRgCSRMatrixVectorProductKernel( Real* target,
                                                        const Real* vect,
                                                        const Real* matrxValues,
                                                        const Index* matrxColumni,
                                                        const Index* _blockInfo,
                                                        const Index* _threadsInfo, 
                                                        const Index numBlocks )
{

	__shared__ Real partialSums[256];
	__shared__ Index info[4];			// first row index, number of rows assigned to the block, number of "rounds", first value and col index
	__shared__ Index threadsInfo[129];

	Index idx, begin, end, column;
	Real vectVal, sum;

	for(Index bId = blockIdx.x; bId < numBlocks; bId += gridDim.x) {
		if(threadIdx.x < 4) {
			info[threadIdx.x] = _blockInfo[4*bId + threadIdx.x];
		}
		__syncthreads();

		if(threadIdx.x < info[1]) {
			threadsInfo[threadIdx.x] = _threadsInfo[info[0]+threadIdx.x];
		}
		if(threadIdx.x == info[1]) {
			threadsInfo[threadIdx.x] = blockDim.x;
		}
		
		sum = 0;
		for(Index i = 0; i<info[2]; i++) {
			idx = threadIdx.x + i*blockDim.x + info[3];
			column = matrxColumni[idx];
			if(column != -1) {
				vectVal = vect[column];
				sum += matrxValues[idx] * vectVal;
			}
		}
		partialSums[threadIdx.x] = sum;
		__syncthreads();

		if(threadIdx.x < info[1]) {
			sum = 0;
			begin = threadsInfo[threadIdx.x];
			end = threadsInfo[threadIdx.x+1];
			for(Index i = begin; i<end; i++) {
				sum += partialSums[i];
			}
			target[info[0]+threadIdx.x] = sum;
		}
	}
}

#endif // ifdef HAVE_CUDA


#endif /* TNLRgCSRMATRIX_H_ */
