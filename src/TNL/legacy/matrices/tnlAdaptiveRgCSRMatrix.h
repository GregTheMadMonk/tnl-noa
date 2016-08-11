/***************************************************************************
                          tnlAdaptiveRgCSR.h  -  description
                             -------------------
    begin                : Mar 19, 2011
    copyright            : (C) 2011 by Martin Heller
    email                : hellemar@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


#ifndef TNLARgCSRMATRIX_H_
#define TNLARgCSRMATRIX_H_

#include <iostream>
#include <iomanip>
#include <TNL/Containers/Vector.h>
#include <TNL/Assert.h>
#include <TNL/core/mfuncs.h>
#include <TNL/Matrices/CSR.h>
#include <TNL/debug/tnlDebug.h>
#include <TNL/core/tnlDevice.h>

using namespace std;

struct tnlARGCSRGroupProperties
{
   int size;
   int chunkSize;
   int firstRow;
   int offset;

   static String getType()
   {
      return String( "tnlARGCSRGroupProperties" );
   };
};

ostream& operator << ( std::ostream& str, const tnlARGCSRGroupProperties& p ){};

inline String getType( const tnlARGCSRGroupProperties& a )
{
   return String( "tnlARGCSRGroupProperties" );
}

//! Matrix storing the non-zero elements in the Row-grouped CSR (Compressed Sparse Row) format
/*!
 */
template< typename Real, typename Device = Devices::Host, typename Index = int >
class tnlAdaptiveRgCSR : public Matrix< Real, Device, Index >
{
   public:
   //! Basic constructor
   tnlAdaptiveRgCSR( const String& name );

   const String& getMatrixClass() const;

   String getType() const;

   Index getMaxGroupSize() const;

   Index getCUDABlockSize() const;

   //! Sets the number of row and columns.
   bool setSize( Index new_size );

   //! Allocate memory for the nonzero elements.
   bool setNonzeroElements( Index elements );

   void reset();

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
                    const Vector< Real, Device, Index >& vec ) const
   { abort(); };

   void vectorProduct( const Vector< Real, Device, Index >& vec,
                       Vector< Real, Device, Index >& result ) const;

   /****
    * This method sets parameters of the format.
    * If it is called after method copyFrom, the matrix will be broken.
    * TODO: Add state ensuring that this situation will lead to error.
    */
   void tuneFormat( const Index desiredChunkSize,
                    const Index cudaBlockSize );

   bool copyFrom( const CSR< Real, Devices::Host,Index >& csr_matrix );

   template< typename Device2 >
   bool copyFrom( const tnlAdaptiveRgCSR< Real, Device2, Index >& rgCSR );

   Real getRowL1Norm( Index row ) const
   { abort(); };

   void multiplyRow( Index row, const Real& value )
   { abort(); };

   //! Prints out the matrix structure
   void printOut( std::ostream& str,
                  const String& format,
		            const Index lines = 0 ) const;

   bool draw( std::ostream& str,
              const String& format,
              CSR< Real, Device, Index >* csrMatrix,
              int verbose = 0 );


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

   /****
    * Returns ID of the first thread mapped to the row of matrix.
    */
   Index getFirstThreadInRow( const Index row, const Index groupId ) const;

   /****
    * Returns ( ID of the last thread mapped to this row of matrix ) + 1
    */
   Index getLastThreadInRow( const Index row, const Index groupId ) const;

   void printOutGroup( std::ostream& str,
                       const Index groupId ) const;

   Vector< Real, Device, Index > nonzeroElements;

   Vector< Index, Device, Index > columns;

   Vector< Index, Device, Index > threads;

   Vector< tnlARGCSRGroupProperties, Device, Index > groupInfo;

   Vector< Index, Device, Index > rowToGroupMapping;

   Index maxGroupSize, groupSizeStep;

   Index desiredChunkSize;

   Index numberOfGroups;

   Index cudaBlockSize;

   Index artificialZeros;

   //! The last non-zero element is at the position last_non_zero_element - 1
   Index lastNonzeroElement;

   friend class tnlAdaptiveRgCSR< Real, Devices::Host, Index >;
   friend class tnlAdaptiveRgCSR< Real, Devices::Cuda, Index >;
};

#ifdef HAVE_CUDA

template< class Real, typename Index >
__global__ void AdaptiveRgCSRVectorProductKernel( Real* target,
                                                        const Real* vect,
                                                        const Real* nonzeroElements,
                                                        const Index* columns,
                                                        const tnlARGCSRGroupProperties* globalGroupInfo,
                                                        const Index* globalThreadsMapping,
                                                        const Index numBlocks );

#endif


template< typename Real, typename Device, typename Index >
tnlAdaptiveRgCSR< Real, Device, Index > :: tnlAdaptiveRgCSR( const String& name )
: Matrix< Real, Device, Index >( name ),
  nonzeroElements( name + " : nonzeroElements" ),
  columns( name + " : columns" ),
  threads( name + " : threads" ),
  groupInfo( name + ": groupInfo" ),
  rowToGroupMapping( name + " : rowToGroupMapping" ),
  maxGroupSize( 16 ),
  groupSizeStep( 16 ),
  desiredChunkSize( 4 ),
  numberOfGroups( 0 ),
  cudaBlockSize( 32 ),
  artificialZeros( 0 ),
  lastNonzeroElement( 0 )
{
	Assert( maxGroupSize > 0, );
};

template< typename Real, typename Device, typename Index >
const String& tnlAdaptiveRgCSR< Real, Device, Index > :: getMatrixClass() const
{
   return MatrixClass :: main;
};

template< typename Real, typename Device, typename Index >
String tnlAdaptiveRgCSR< Real, Device, Index > :: getType() const
{
   return String( "tnlAdaptiveRgCSR< ") +
          String( getType( Real( 0.0 ) ) ) +
          String( ", " ) +
          Device :: getDeviceType() +
          String( ", " ) +
          getType( Index( 0 ) ) +
          String( " >" );
};

template< typename Real, typename Device, typename Index >
Index tnlAdaptiveRgCSR< Real, Device, Index > :: getMaxGroupSize() const
{
   return maxGroupSize;
}

template< typename Real, typename Device, typename Index >
Index tnlAdaptiveRgCSR< Real, Device, Index > :: getCUDABlockSize() const
{
   return cudaBlockSize;
}

template< typename Real, typename Device, typename Index >
bool tnlAdaptiveRgCSR< Real, Device, Index > :: setSize( Index newSize )
{
   Assert( newSize > 0, std::cerr << "newSize = " << newSize );
   this->size = newSize;
   if( ! groupInfo. setSize( this->getSize() ) ||
       ! threads. setSize( this->getSize() ) ||
       ! rowToGroupMapping. setSize( this->getSize() ) )
      return false;
   threads. setValue( 0 );
   rowToGroupMapping. setValue( 0 );
   lastNonzeroElement = 0;
   return true;
};

template< typename Real, typename Device, typename Index >
bool tnlAdaptiveRgCSR< Real, Device, Index > :: setNonzeroElements( Index elements )
{
   Assert( elements != 0, );
   if( ! nonzeroElements. setSize( elements ) ||
       ! columns. setSize( elements ) )
      return false;
   nonzeroElements. setValue( 0.0 );
   columns. setValue( -1 );
   return true;
};


template< typename Real, typename Device, typename Index >
void tnlAdaptiveRgCSR< Real, Device, Index > :: reset()
{
   nonzeroElements. reset();
   columns. reset();
   threads. reset();
   groupInfo. reset();
   rowToGroupMapping. reset();
   maxGroupSize = 16;
   groupSizeStep = 16;
   desiredChunkSize = 4;
   numberOfGroups = 0;
   cudaBlockSize = 32;
   artificialZeros = 0;
   lastNonzeroElement = 0;
};

template< typename Real, typename Device, typename Index >
Index tnlAdaptiveRgCSR< Real, Device, Index > :: getNonzeroElements() const
{
   Assert( nonzeroElements. getSize() > artificialZeros, );
	return nonzeroElements. getSize() - artificialZeros;
}

template< typename Real, typename Device, typename Index >
Index tnlAdaptiveRgCSR< Real, Device, Index > :: getArtificialZeroElements() const
{
	return artificialZeros;
}

template< typename Real, typename Device, typename Index >
void tnlAdaptiveRgCSR< Real, Device, Index > :: tuneFormat( const Index desiredChunkSize,
                                                                  const Index cudaBlockSize )
{
   this->desiredChunkSize = desiredChunkSize;
   this->cudaBlockSize = cudaBlockSize;
}

template< typename Real, typename Device, typename Index >
Index tnlAdaptiveRgCSR< Real, Device, Index > :: getFirstThreadInRow( const Index row, const Index groupId ) const
{
   dbgFunctionName( "tnlAdaptiveRgCSR< Real, Devices::Host >", "getFirstThreadInRow" );
   Assert( row >= 0 && row < this->getSize(), std::cerr << " row = " << row << " size = " << this->getSize() );
   //dbgExpr( row );
   //dbgExpr( groupInfo[ groupId ]. firstRow );
   if( row == groupInfo[ groupId ]. firstRow )
      return 0;
   return threads. getElement( row - 1 );
}

template< typename Real, typename Device, typename Index >
Index tnlAdaptiveRgCSR< Real, Device, Index > :: getLastThreadInRow( const Index row, const Index groupId ) const
{
   Assert( row >= 0 && row < this->getSize(), std::cerr << " row = " << row << " size = " << this->getSize() );
   return threads. getElement( row );
}

template< typename Real, typename Device, typename Index >
bool tnlAdaptiveRgCSR< Real, Device, Index > :: copyFrom( const CSR< Real, Devices::Host, Index >& csrMatrix )
{
  dbgFunctionName( "tnlAdaptiveRgCSR< Real, Devices::Host >", "copyFrom" );
  if( ! this->setSize( csrMatrix. getSize() ) )
          return false;
 
  if( Device :: getDevice() == Devices::HostDevice )
  {
      Index nonzerosInGroup( 0 );
      Index groupBegin( 0 );
      Index groupEnd( 0 );
      Index rowsInGroup( 0 );
      Index groupId( 0 );

      Index numberOfStoredValues( 0 );
      Vector< Index, Devices::Host, Index > threadsPerRow( "tnlAdaptiveRgCSR< Real, Device, Index > :: copyFrom : threadsPerRow" );
      threadsPerRow. setSize( cudaBlockSize );
      threadsPerRow. setValue( 0 );

      /****
       * This loop computes sizes of the groups and the number of threads per one row
       */
      while( true )
      {
         /****
          * First compute the group size such that the number of the non-zero elements in each group is
          * approximately the same.
          */
         groupEnd += this->groupSizeStep;
         groupEnd = min( groupEnd, this->getSize() );

         nonzerosInGroup = csrMatrix. row_offsets[ groupEnd ] - csrMatrix. row_offsets[ groupBegin ];
         rowsInGroup = groupEnd - groupBegin;

         if( nonzerosInGroup < cudaBlockSize * desiredChunkSize &&
             groupEnd < this->getSize() &&
             rowsInGroup < cudaBlockSize )
            continue;

         dbgCout( " groupBegin = " << groupBegin
                  << " groupEnd = " << groupEnd
                  << " nonzerosInGroup = " << nonzerosInGroup );

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
            double nonzerosInRow = csrMatrix. getNonzeroElementsInRow( i );
            double nonzerosInRowRatio( 0.0 );
            if( nonzerosInGroup != 0.0 )
               nonzerosInRowRatio = nonzerosInRow / ( double ) nonzerosInGroup;
            usedThreads += threadsPerRow[ i - groupBegin ] = max( 1.0, floor( freeThreads * nonzerosInRowRatio ) );
         }
         /****
          * If there are some threads left distribute them to the rows from the group beginning.
          * TODO: add the free threads to the longest rows
          *  - find row with the largest chunks and add one thread to this row
          *  - repeat it
          */
         Index threadsLeft = cudaBlockSize - usedThreads;
         dbgExpr( usedThreads );
         dbgExpr( threadsLeft );
         //for( Index i = 0; i < threadsLeft; i++)
         //   threadsPerRow[ i % rowsInGroup ] ++;
         while( usedThreads < cudaBlockSize )
         {
            Index maxChunkSize( 0 );
            for( Index row = groupBegin; row < groupEnd; row ++ )
            {
               double nonzerosInRow = csrMatrix. getNonzeroElementsInRow( row );
               Index chunkSize( 0 );
               if( threadsPerRow[ row - groupBegin ] != 0 )
                  chunkSize = ceil( nonzerosInRow / ( double ) threadsPerRow[ row - groupBegin ] );
               maxChunkSize = max( chunkSize, maxChunkSize );
            }
            for( Index row = groupBegin; row < groupEnd; row ++ )
            {
               double nonzerosInRow = csrMatrix. getNonzeroElementsInRow( row );
               Index chunkSize( 0 );
               if( threadsPerRow[ row - groupBegin ] != 0 )
                  chunkSize = ceil( nonzerosInRow / ( double ) threadsPerRow[ row - groupBegin ] );
               if( chunkSize == maxChunkSize && usedThreads < cudaBlockSize )
               {
                  threadsPerRow[ row - groupBegin ] ++;
                  usedThreads ++;
               }
            }
         }

         /****
          * Compute prefix-sum on threadsPerRow and store it in threads
          */
         threads[ groupBegin ] = threadsPerRow[ 0 ];
         dbgExpr( threads[ groupBegin ] );
         for( Index i = groupBegin + 1; i< groupEnd; i++ )
         {
            threads[ i ] = threads[ i - 1 ] + threadsPerRow[ i - groupBegin ];
            dbgExpr( threads[ i ] );
         }

         /****
          * Now, compute the chunk size
          */
         Index maxChunkSize( 0 );
         for( Index i = groupBegin; i < groupEnd; i++ )
         {
            double nonzerosInRow = csrMatrix. getNonzeroElementsInRow( i );
            const Index chunkSize = ceil( nonzerosInRow / ( double ) threadsPerRow[ i - groupBegin ] );
            maxChunkSize = max( chunkSize, maxChunkSize );
         }
         groupInfo[ groupId ]. size = rowsInGroup;
         groupInfo[ groupId ]. firstRow = groupBegin;
         groupInfo[ groupId ]. offset = numberOfStoredValues;
         groupInfo[ groupId ]. chunkSize = maxChunkSize;

         dbgCout( "New group: Id = " << groupId
                  << " size = " << groupInfo[ groupId ]. size
                  << " first row = " << groupInfo[ groupId ]. firstRow
                  << " offset = " << groupInfo[ groupId ]. offset
                  << " chunk size = " << groupInfo[ groupId ]. chunkSize );

         for( Index i = groupBegin; i < groupEnd; i ++ )
            rowToGroupMapping[ i ] = groupId;

         groupId++;
         numberOfStoredValues += cudaBlockSize * maxChunkSize;
         groupBegin = groupEnd;

         if( groupBegin == this->getSize() )
         {
            numberOfGroups = groupId;
            break;
         }
      }

      /****
       * Allocate the non-zero elements (they contains some artificial zeros.)
       */
      dbgCout( "Allocating " << max( 1, numberOfStoredValues ) << " elements.");
      if( ! setNonzeroElements( max( 1, numberOfStoredValues ) ) )
         return false;
      artificialZeros = numberOfStoredValues - csrMatrix. getNonzeroElements();

      lastNonzeroElement = numberOfStoredValues;

      dbgCout( "Inserting data " );

      Index index, baseRow;
      for( Index groupId = 0; groupId < numberOfGroups; groupId ++ )
      {
          dbgCout( "Inserting to group: Id = " << groupId
                     << " size = " << groupInfo[ groupId ]. size
                     << " first row = " << groupInfo[ groupId ]. firstRow
                     << " offset = " << groupInfo[ groupId ]. offset
                     << " chunk size = " << groupInfo[ groupId ]. chunkSize );

         baseRow = groupInfo[ groupId ]. firstRow;
         index = groupInfo[ groupId ]. offset;

         /****
          * Now do the insertion
          */
         for( Index groupRow = 0; groupRow < groupInfo[ groupId ]. size; groupRow ++ )
         {
            const Index matrixRow = groupRow + baseRow;
            dbgCout( "Row = " << matrixRow <<
                     " group row = " << groupRow <<
                     " firstThreadInRow = " << this->getFirstThreadInRow( matrixRow, groupId ) <<
                     " lastThreadInRow = " << this->getLastThreadInRow( matrixRow, groupId ) <<
                     " inserting offset = " << index );
            Index pos = csrMatrix. row_offsets[ matrixRow ];
            Index rowCounter( 0 );
            for( Index thread = this->getFirstThreadInRow( matrixRow, groupId );
                 thread < this->getLastThreadInRow( matrixRow, groupId );
                 thread ++ )
            {
               Index insertPosition = groupInfo[ groupId ]. offset + thread;
               for( Index k = 0; k < groupInfo[ groupId ]. chunkSize; k ++ )
               {
                  Assert( index < numberOfStoredValues, std::cerr << "Index = " << index << " numberOfStoredValues = " << numberOfStoredValues );
                  if( rowCounter < csrMatrix. getNonzeroElementsInRow( matrixRow ) )
                  {
                     dbgCout( "Inserting data from CSR format at position " << pos << " to AdaptiveRgCSR at " << insertPosition );
                     nonzeroElements[ insertPosition ] = csrMatrix. nonzero_elements[ pos ];
                     columns[ insertPosition ] = csrMatrix. columns[ pos ];
                     pos ++;
                  }
                  else
                  {
                     //dbgCout( "Inserting artificial zero to AdaptiveRgCSR at " << index );
                     columns[ insertPosition ] = -1;
                     nonzeroElements[ insertPosition ] = 0.0;
                  }
                  insertPosition += cudaBlockSize;
                  rowCounter ++;
               }
            }
         }

         /*
         Vector< Index, Devices::Host, Index > counters( "tnlAdaptiveRgCSR< Real, Device, Index > :: copyFrom : counters" );
         counters. setSize( cudaBlockSize );
         counters. setValue( 0 );
         for( Index k = 0; k < groupInfo[ groupId ]. chunkSize; k ++ )
            for( Index row = 0; row < groupInfo[ groupId ]. size; row ++ )
            {
               const Index matrixRow = groupRow + baseRow;
               dbgCout( "group row = " << row <<
                        " firstThreadInRow = " << this->getFirstThreadInRow( matrixRow, groupId ) <<
                        " lastThreadInRow = " << this->getLastThreadInRow( matrixRow, groupId ) <<
                        " inserting offset = " << index );
               for( Index thread = this->getFirstThreadInRow( matrixRow, groupId );
                    thread < this->getLastThreadInRow( matrixRow, groupId );
                    thread ++ )
               {
                  Assert( index < numberOfStoredValues, std::cerr << "Index = " << index << " numberOfStoredValues = " << numberOfStoredValues );
                  if( counters[ row ] < csrMatrix. getNonzeroElementsInRow( matrixRow ) )
                  {
                     Index pos = csrMatrix. row_offsets[ matrixRow ] + counters[ row ];
                     //dbgCout( "Inserting data from CSR format at position " << pos << " to AdaptiveRgCSR at " << index );
                     nonzeroElements[ index ] = csrMatrix. nonzero_elements[ pos ];
                     columns[ index ] = csrMatrix. columns[ pos ];
                  }
                  else
                  {
                     //dbgCout( "Inserting artificial zero to AdaptiveRgCSR at " << index );
                     columns[ index ] = -1;
                     nonzeroElements[ index ] = 0.0;
                  }
                  counters[ row ] ++;
                  index ++;
               }
            }*/

      }
	}
	if( Device :: getDevice() == Devices::CudaDevice )
	{
		Assert( false,
			cerr << "Conversion from CSR on the host to the tnlAdaptiveRgCSR on the CUDA device is not implemented yet."; );
		//TODO: implement this
	}
	return true;
}


template< typename Real, typename Device, typename Index >
   template< typename Device2 >
bool tnlAdaptiveRgCSR< Real, Device, Index > :: copyFrom( const tnlAdaptiveRgCSR< Real, Device2, Index >& adaptiveRgCSR )
{
   dbgFunctionName( "tnlAdaptiveRgCSR< Real, Device, Index >", "copyFrom" );
   maxGroupSize = adaptiveRgCSR. maxGroupSize;
   groupSizeStep = adaptiveRgCSR. groupSizeStep;
   desiredChunkSize = adaptiveRgCSR. desiredChunkSize;
   cudaBlockSize = adaptiveRgCSR. cudaBlockSize;
   lastNonzeroElement = adaptiveRgCSR. lastNonzeroElement;
   numberOfGroups = adaptiveRgCSR. numberOfGroups;
 

   if( ! this->setSize( adaptiveRgCSR. getSize() ) )
      return false;

   /****
    * Allocate the non-zero elements (they contains some artificial zeros.)
    */
   Index total_elements = adaptiveRgCSR. getNonzeroElements() +
                          adaptiveRgCSR. getArtificialZeroElements() ;
   dbgCout( "Allocating " << total_elements << " elements.");
   if( ! setNonzeroElements( total_elements ) )
      return false;
   artificialZeros = total_elements - adaptiveRgCSR. getNonzeroElements();

   nonzeroElements = adaptiveRgCSR. nonzeroElements;
   columns = adaptiveRgCSR. columns;
   groupInfo = adaptiveRgCSR. groupInfo;
   threads = adaptiveRgCSR. threads;
   rowToGroupMapping = adaptiveRgCSR. rowToGroupMapping;

   return true;
};

template< typename Real, typename Device, typename Index >
Real tnlAdaptiveRgCSR< Real, Device, Index > :: getElement( Index row,
                                                                  Index column ) const
{
   dbgFunctionName( "tnlAdaptiveRgCSR< Real, Devices::Host >", "getElement" );
   Assert( 0 <= row && row < this->getSize(),
              std::cerr << "The row is outside the matrix." );
   if( Device :: getDevice() == Devices::HostDevice )
   {
      const Index groupId = rowToGroupMapping[ row ];
      const Index firstRow = groupInfo[ groupId ]. firstRow;
      const Index lastRow = firstRow + groupInfo[ groupId ]. size;
      Index pointer = groupInfo[ groupId ]. offset;

      for( Index chunkOffset = 0; chunkOffset < groupInfo[ groupId ]. chunkSize; chunkOffset ++ )
         for( Index currentRow = firstRow; currentRow < lastRow; currentRow ++ )
         {
            if( currentRow != row )
               pointer += this->getLastThreadInRow( currentRow, groupId ) - this->getFirstThreadInRow( currentRow, groupId );
            else
               for( Index i = this->getFirstThreadInRow( currentRow, groupId );
                    i < this->getLastThreadInRow( currentRow, groupId );
                    i ++ )
               {
                  if( columns[ pointer ] == column )
                     return nonzeroElements[ pointer ];
                  pointer ++;
               }
         }
      return 0.0;
   }
   if( Device :: getDevice() == Devices::CudaDevice )
   {
      Assert( false,
                std::cerr << "tnlRgCSR< Real, Devices::Cuda, Index > ::getElement is not implemented yet." );
      //TODO: implement this

   }
   return 0.0; // this is only to avoid a compiler warning
}

template< typename Real, typename Device, typename Index >
void tnlAdaptiveRgCSR< Real, Device, Index > :: vectorProduct( const Vector< Real, Device, Index >& vec,
                                                                     Vector< Real, Device, Index >& result ) const
{
   dbgFunctionName( "tnlAdaptiveRgCSR< Real, Devices::Host >", "vectorProduct" )
   Assert( vec. getSize() == this->getSize(),
              std::cerr << "The matrix and vector for a multiplication have different sizes. "
                   << "The matrix size is " << this->getSize() << "."
                   << "The vector size is " << vec. getSize() << std::endl; );
   Assert( result. getSize() == this->getSize(),
              std::cerr << "The matrix and result vector of a multiplication have different sizes. "
                   << "The matrix size is " << this->getSize() << "."
                   << "The vector size is " << vec. getSize() << std::endl; );

   if( Device :: getDevice() == Devices::HostDevice )
   {

      Real partialSums[ 256 ];
      const Index blockDim = this->getCUDABlockSize();
      for( Index bId = 0; bId < numberOfGroups; bId ++ )
      //for( Index bId = 0; bId < 1; bId ++ )
      {
         const Index groupId = bId;
         //printOutGroup(std::cout, bId );
         for( Index threadIdx = 0; threadIdx < blockDim; threadIdx ++ )
         {

            /****
             * Each thread now computes partial sum in its chunk
             */
            Real sum = 0;
            for( Index i = 0; i < groupInfo[ bId ]. chunkSize; i ++ )
            {
               const Index offset = threadIdx + i * blockDim + groupInfo[ bId ]. offset;
               const Index column = columns[ offset ];
               if( column != -1 )
               {
                  sum += nonzeroElements[ offset ] * vec[ column ];
                  //cout << "A. Chunk = " << threadIdx << " Value = " << std::setprecision( 10 ) << nonzeroElements[ offset ] << std::endl;
               }
            }
            partialSums[ threadIdx ] = sum;
            dbgCout( "partialSums[ " << threadIdx << " ] = " << partialSums[ threadIdx ] );
            //cout << "partialSums[ " << threadIdx << " ] = " << partialSums[ threadIdx ] << std::endl;
         }
         /****
          * Now sum the partial sums in each row
          */
         for( Index threadIdx = 0; threadIdx < blockDim; threadIdx ++ )
         {
            if( threadIdx < groupInfo[ bId ]. size )
            {
               const Index row = groupInfo[ bId ]. firstRow + threadIdx;
               Index firstChunk = getFirstThreadInRow( row, groupId );
               Index lastChunk = getLastThreadInRow( row, groupId );
               dbgCout( "firstChunk = " << firstChunk << " lastChunk = " << lastChunk );

               Real sum = 0;
               for( Index i = firstChunk; i < lastChunk; i++ )
                  sum += partialSums[ i ];
               result[ row ] = sum;

#ifdef TNLARgCSRMATRIX_CHECK_SPMV
               /****
                * Check the partial sums
                */
               if( ! groupInfo[ bId ]. chunkSize )
                  continue;
               Index rowCounter( 0 ), chunkCounter( firstChunk );
               Real partialSum( 0.0 );
               for( Index j = 0; j < this->getSize(); j ++)
               {
                  const Real val = this->getElement( row, j );
                  if( val != 0 )
                  {
                     if( row == 2265 )
                        std::cerr << "A. col = " << j << " val = " << val << std::endl;
                     partialSum += val * vec[ j ];
                     rowCounter ++;
                     //cout << "B. Chunk = " << chunkCounter << " Value = " << std::setprecision( 10 ) << val << std::endl;
                     if( rowCounter % groupInfo[ bId ]. chunkSize == 0 )
                     {
                        if( chunkCounter >= lastChunk )
                           std::cerr << "I found more chunks ( ID. " << chunkCounter << " ) than I expected ( max. ID " << lastChunk << ") on the line " << row << std::endl;
                        if( partialSum != partialSums[ chunkCounter ] )
                        {
                           std::cerr << "Partial sum error: row = " << row
                                << " chunk = " << chunkCounter
                                << " partialSums[ " << chunkCounter << " ] = " << partialSums[ chunkCounter ]
                                << " partialSum = " << partialSum << std::endl;
                           partialSums[ chunkCounter ] = partialSum;
                        }
                        chunkCounter ++;
                        partialSum = 0;
                     }
                  }
               }
               if( partialSum )
               {
                  if( partialSum != partialSums[ chunkCounter ] )
                  {
                     std::cerr << "Partial sum error: row = " << row
                          << " chunk = " << chunkCounter
                          << " partialSums[ " << chunkCounter << " ] = " << partialSums[ chunkCounter ]
                          << " partialSum = " << partialSum << std::endl;
                     partialSums[ chunkCounter ] = partialSum;
                  }
                  chunkCounter ++;
               }
               if( chunkCounter < lastChunk - 1 )
               {
                  std::cerr << "I found wrong number of chunks ( ID. " << chunkCounter << " ) than I expected ( max. ID " << lastChunk << ") on the line " << row << std::endl;
                  for( Index i = chunkCounter; i < lastChunk; i ++ )
                  {
                     std::cerr << "   partialSums[ " << i << " ] = " << partialSums[ i ] << std::endl;
                     //partialSums[ i ] = 0.0;
                  }
               }


               /****
                * Check the result with the method getElement
                */
               Real checkSum( 0.0 );
               for( Index i = 0; i < this->getSize(); i ++ )
                  checkSum += this->getElement( row, i );// * vec[ i ];

               if( checkSum != sum )
               {
                  std::cerr << "row = " << row << " sum = " << sum << " checkSum = " << checkSum << " diff = " << sum - checkSum << std::endl;
                  //result[ row ] = checkSum;
               }

               //cerr << "result[" << row << "] = " << result[ row ] << std::endl;
#endif // TNLARgCSRMATRIX_CHECK_SPMV
            }
         }
      }

#ifdef UNDEF
      Index idx[ TB_SIZE ];
      Real psum[ TB_SIZE ];        //partial sums for each thread
      Index limits[ MAX_ROWS + 1 ];  //indices of first threads for each row + index of first unused thread
      Real results[ MAX_ROWS ];

      /****
       * Go over all groups ...
       */
      dbgExpr( this->numberOfGroups );
      for( Index groupId = 0; groupId < this->numberOfGroups; groupId ++ )
      {
         /****
          * In each group compute partial sums of each thread
          */
         dbgExpr( groupId );
         for( Index thread = 0; thread < cudaBlockSize; thread ++ )
         {
            idx[ thread ] = this->groupInfo[ groupId ]. offset + thread;
            psum[ thread ] = 0;
            for( Index chunkOffset = 0;
                 chunkOffset < this->groupInfo[ groupId ]. chunkSize;
                 chunkOffset ++ )
            {
               if( this->columns[ idx[ thread ] ] != -1  )
                  psum[ thread ] += this->nonzeroElements[ idx[ thread ] ] * vec[ this->columns[ idx[ thread ] ] ];
               idx[ thread ] += cudaBlockSize;
            }
            dbgExpr( psum[ thread ] );
         }

         /****
          * Compute reduction over threads in each row
          */
         for( Index row = groupInfo[ groupId ]. firstRow;
              row < groupInfo[ groupId ]. firstRow + groupInfo[ groupId ]. size;
              row ++ )
         {
            dbgCout( "Row: " << row << " firstThreadInRow: " << this->getFirstThreadInRow( row, groupId ) << " lastThreadInRow: " << this->getLastThreadInRow( row, groupId ) );
            result[ row ] = 0.0;
            for( Index thread = this->getFirstThreadInRow( row, groupId );
                 thread < this->getLastThreadInRow( row, groupId );
                 thread ++ )
            {
               result[ row ] += psum[ thread ];
               dbgCout( "Thread: " << thread << " psum[ thread ]: " << psum[ thread ] << " result[ row ]: " << result[ row ] );
            }
         }
      }
#endif
   }
   if( Device :: getDevice() == Devices::CudaDevice )
   {
#ifdef HAVE_CUDA
   Index blockSize = this->getCUDABlockSize();
   const Index size = this->getSize();

   Index desGridSize;
	desGridSize = this->numberOfGroups;
	//desGridSize = (desGridSize < 4096) ? desGridSize : 4096;

   cudaThreadSynchronize();
   int gridSize = (int) desGridSize;
   dim3 gridDim( gridSize ), blockDim( blockSize );

   //cerr << "gridSize = " << gridDim. x << std::endl;
   //cerr << "blockSize = " << blockDim. x << std::endl;
   size_t allocatedSharedMemory = blockDim. x * sizeof( Real ) +
                                  sizeof( tnlARGCSRGroupProperties ) +
                                  blockDim. x * sizeof( int );

   cudaThreadSetCacheConfig( cudaFuncCachePreferL1 );
   AdaptiveRgCSRVectorProductKernel< Real, Index >
                                            <<< gridDim, blockDim, allocatedSharedMemory >>>
                                            ( result. getData(),
                                              vec. getData(),
                                              nonzeroElements. getData(),
                                              columns. getData(),
                                              groupInfo. getData(),
                                              threads. getData(),
                                              1 );
    cudaThreadSynchronize();
    CHECK_CUDA_ERROR;
#else
    CudaSupportMissingMessage;;
#endif
   }

}

template< typename Real, typename Device, typename Index >
void tnlAdaptiveRgCSR< Real, Device, Index > :: printOutGroup( std::ostream& str,
                                                                     const Index groupId ) const
{
   const Index firstRow = groupInfo[ groupId ]. firstRow;
   const Index lastRow = firstRow + groupInfo[ groupId ]. size;
   str << std::endl << "Group number: " << groupId << std::endl;
   str << " Rows: " << firstRow << " -- " << lastRow << std::endl;
   str << " Chunk size: " << groupInfo[ groupId ]. chunkSize << std::endl;
   str << " Threads mapping: ";
   for( Index row = firstRow; row < lastRow; row ++ )
      str << threads. getElement( row ) << "  ";
   str << std::endl;
   str << " Group offset: " << groupInfo[ groupId ]. offset <<  std::endl;
   Index pointer = groupInfo[ groupId ]. offset;
   Index groupBaseRow = groupInfo[ groupId ]. firstRow;
   for( Index row = firstRow; row < lastRow; row ++ )
   {
      Index firstThread = this->getFirstThreadInRow( row, groupId );
      Index lastThread = this->getLastThreadInRow( row, groupId );
      str << " Row number: " << row << " Threads: " << firstThread << " -- " << lastThread << std::endl;
      for( Index thread = firstThread; thread < lastThread; thread ++ )
      {
         Index threadOffset = this->groupInfo[ groupId ]. offset + thread;
         str << "  Thread: " << thread << " Thread Offset: " << threadOffset << " Chunk: ";
         for( Index i = 0; i < groupInfo[ groupId ]. chunkSize; i ++ )
            str << this->nonzeroElements[ threadOffset + i * cudaBlockSize ] << "["
                << this->columns[ threadOffset + i * cudaBlockSize ] << "], ";
         str << std::endl;
      }
      str << std::endl;
   }
}


template< typename Real, typename Device, typename Index >
void tnlAdaptiveRgCSR< Real, Device, Index > :: printOut( std::ostream& str,
                                                                const String& name,
                                                                const String& format,
		                                                          const Index lines ) const
{
   if( format == "" || format == "text" )
   {
      str << "Structure of tnlAdaptiveRgCSR" << std::endl;
      str << "Matrix name:" << name << std::endl;
      str << "Matrix size:" << this->getSize() << std::endl;
      str << "Allocated elements:" << nonzeroElements. getSize() << std::endl;
      str << "Number of groups: " << numberOfGroups << std::endl;

      Index print_lines = lines;
      if( ! print_lines )
         print_lines = this->getSize();

      for( Index groupId = 0; groupId < numberOfGroups; groupId ++ )
      {
         const Index firstRow = groupInfo[ groupId ]. firstRow;
         if( firstRow  > print_lines )
            return;
         printOutGroup( str, groupId );
      }

      str << std::endl;
   }
   if( format == "html" )
   {
      str << "<h1>Structure of tnlAdaptiveRgCSR</h1>" << std::endl;
      str << "<b>Matrix name:</b> " << name << "<p>" << std::endl;
      str << "<b>Matrix size:</b> " << this->getSize() << "<p>" << std::endl;
      str << "<b>Allocated elements:</b> " << nonzeroElements. getSize() << "<p>" << std::endl;
      str << "<b>Number of groups:</b> " << this->numberOfGroups << "<p>" << std::endl;
      str << "<table border=1>" << std::endl;
      str << "<tr> <td> <b> GroupId </b> </td> <td> <b> Size </b> </td> <td> <b> Chunk size </b> </td> <td> <b> % of nonzeros </b> </td> </tr>" << std::endl;
      Index print_lines = lines;
      if( ! print_lines )
         print_lines = this->getSize();

      Index minGroupSize( this->getSize() );
      Index maxGroupSize( 0 );
      for( Index i = 0; i < this->numberOfGroups; i ++ )
      {
         const Index groupSize = this->groupInfo. getElement( i ). size;
         minGroupSize = min( groupSize, minGroupSize );
         maxGroupSize = max( groupSize, maxGroupSize );
         const Index chunkSize = this->groupInfo. getElement( i ). chunkSize;
         const Index allElements = chunkSize * this->cudaBlockSize;
         double filling = ( double ) ( allElements ) /
                          ( double ) this->nonzeroElements. getSize();
         str << "<tr> <td> " << i
            << "</td> <td> " << groupSize
            << "</td> <td> " << chunkSize
            << " </td> <td> " << 100.0 * filling << "% </td></tr>" << std::endl;
      }
      str << "</table>" << std::endl;
      str << "<b> Min. group size:</b> " << minGroupSize << "<p>" << std::endl;
      str << "<b> Max. group size:</b> " << maxGroupSize << "<p>" << std::endl;
      str << "<b> Ratio:</b> " << ( double ) maxGroupSize / ( double ) minGroupSize << std::endl;
      str << std::endl;
   }
};

template< typename Real, typename Device, typename Index >
bool tnlAdaptiveRgCSR< Real, Device, Index > :: draw( std::ostream& str,
                                                            const String& format,
                                                            CSR< Real, Device, Index >* csrMatrix,
                                                            int verbose )
{
   if( Device :: getDevice() == Devices::CudaDevice )
   {
      std::cerr << "Drawing of matrices stored on the GPU is not supported yet." << std::endl;
      return false;
   }
   if( format == "gnuplot" )
      return Matrix< Real, Device, Index > ::  draw( str, format, csrMatrix, verbose );
   if( format == "eps" )
   {
      const int elementSize = 10;
      this->writePostscriptHeader( str, elementSize );

      /****
       * Draw the groups
       */
      for( Index groupId = 0; groupId < numberOfGroups; groupId ++ )
      {
         const Index groupSize = this->groupInfo. getElement( groupId ). size;
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
}


#ifdef HAVE_CUDA

template< class Real, typename Index >
__global__ void AdaptiveRgCSRVectorProductKernel( Real* target,
                                                        const Real* vect,
                                                        const Real* nonzeroElements,
                                                        const Index* columns,
                                                        const tnlARGCSRGroupProperties* globalGroupInfo,
                                                        const Index* globalThreadsMapping,
                                                        const Index numBlocks )
{

   extern __shared__ int sdata[];

   const int* globalGroupInfoPointer = reinterpret_cast< const int* >( globalGroupInfo );
   tnlARGCSRGroupProperties* groupInfo = reinterpret_cast< tnlARGCSRGroupProperties* >( &sdata[ 0 ] );

   Index* threadsMapping = reinterpret_cast< Index* >( &sdata[ 4 ] );

   Real* partialSums = reinterpret_cast< Real* >( &sdata[ 4 + blockDim. x] );

	//for( Index bId = blockIdx.x; bId < numBlocks; bId += gridDim.x)
	{
	   Index bId = blockIdx.x;
	   /****
	    * Read the group info from the global memory
	    */
		if( threadIdx.x < 4 )
			sdata[ threadIdx.x ] = globalGroupInfoPointer[ 4 * bId + threadIdx.x ];
		__syncthreads();
		/*if( threadIdx. x == 0 )
		{
		   printf( "Group ( %d) size = %d \n", bId, groupInfo -> size );
		   printf( "Group ( %d) Chunk size = %d \n", bId, groupInfo -> chunkSize );
		   printf( "Group ( %d) first row = %d \n", bId, groupInfo -> firstRow );
		   printf( "Group ( %d) offset = %d \n", bId, groupInfo -> offset );
		}*/

		/****
		 * Read mapping of threads to rows.
		 * It says IDs of threads that will work on each row.
		 */
		threadsMapping[ threadIdx. x ] = globalThreadsMapping[ groupInfo -> firstRow + threadIdx. x ];

		/****
		 * Each thread now computes partial sum in its chunk
		 */
		Real sum = 0;
		for( Index i = 0; i < groupInfo -> chunkSize; i ++ )
		{
			const Index offset = threadIdx. x + i * blockDim. x + groupInfo -> offset;
			const Index column = columns[ offset ];
			if( column != -1 )
				sum += nonzeroElements[ offset ] * vect[ column ];
			else
			   break;
		}
		partialSums[ threadIdx. x ] = sum;
		__syncthreads();
		//printf( "Thread %d psum = %f \n", threadIdx. x, sum );


		/****
		 * Now sum the partial sums in each row
		 */
		if( threadIdx. x < groupInfo -> size )
		{
			sum = 0;
			Index begin( 0 );
			const Index row = groupInfo -> firstRow + threadIdx. x;
			if( threadIdx. x > 0 )
			   begin = threadsMapping[ threadIdx. x - 1];
			Index end = threadsMapping[ threadIdx. x ];
			//printf( "Row. %d begin = %d end = %d  \n", row, begin, end );
			for( Index i = begin; i < end; i++ )
				sum += partialSums[ i ];

			target[ row ] = sum;
			//printf( "Summing by thread %d sum = %f \n", threadIdx. x, sum );
		}
	}
}

#endif // ifdef HAVE_CUDA


#endif /* TNLARgCSRMATRIX_H_ */
