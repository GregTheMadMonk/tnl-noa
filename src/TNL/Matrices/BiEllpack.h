/***************************************************************************
                          BiEllpack.h  -  description
                             -------------------
    begin                : Aug 27, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/****
 * This class implements BiELL format from:
 * 
 * Zheng C., Gu S., Gu T.-X., Yang B., Liu X.-P., 
 * BiELL: A bisection ELLPACK-based storage format for optimizing SpMV on GPUs,
 * Journal of Parallel and Distributed Computing, 74 (7), pp. 2639-2647, 2014.
 */

#pragma once

#include <TNL/Matrices/Sparse.h>
#include <TNL/Containers/Vector.h>

namespace TNL {
   namespace Matrices {


template< typename Device >
class BiEllpackDeviceDependentCode;

template< typename Real, typename Device = Devices::Cuda, typename Index = int, int StripSize = 32 >
class BiEllpack : public Sparse< Real, Device, Index >
{
public:
	typedef Real RealType;
	typedef Device DeviceType;
	typedef Index IndexType;
	typedef typename Sparse< RealType, DeviceType, IndexType >::CompressedRowLengthsVector CompressedRowLengthsVector;
	typedef typename Sparse< RealType, DeviceType, IndexType >::ValuesVector ValuesVector;
	typedef typename Sparse< RealType, DeviceType, IndexType >::ColumnIndexesVector ColumnIndexesVector;
	typedef BiEllpack< Real, Device, Index > ThisType;
	typedef BiEllpack< Real, Devices::Host, Index > HostType;
	typedef BiEllpack< Real, Devices::Cuda, Index > CudaType;

	BiEllpack();

	static String getType();

	String getTypeVirtual() const;

	void setDimensions( const IndexType rows,
	                    const IndexType columns );

	void setCompressedRowLengths( const CompressedRowLengthsVector& rowLengths );

	IndexType getRowLength( const IndexType row ) const;

	template< typename Real2,
			  typename Device2,
			  typename Index2 >
	bool setLike( const BiEllpack< Real2, Device2, Index2, StripSize >& matrix );

	void getRowLengths( CompressedRowLengthsVector& rowLengths ) const;

	bool setElement( const IndexType row,
					 const IndexType column,
					 const RealType& value );

#ifdef HAVE_CUDA
	__device__ __host__
#endif
	bool setElementFast( const IndexType row,
						 const IndexType column,
						 const RealType& value );

	bool addElement( const IndexType row,
					 const IndexType column,
					 const RealType& value,
					 const RealType& thisElementMultiplicator = 1.0 );

#ifdef HAVE_CUDA
	__device__ __host__
#endif
	bool addElementFast( const IndexType row,
						 const IndexType column,
						 const RealType& value,
						 const RealType& thisElementMultiplicator = 1.0 );

	bool setRow( const IndexType row,
				 const IndexType* columns,
				 const RealType* values,
				 const IndexType numberOfElements );

	bool addRow( const IndexType row,
				 const IndexType* columns,
				 const RealType* values,
				 const IndexType numberOfElements,
				 const RealType& thisElementMultiplicator = 1.0 );

	RealType getElement( const IndexType row,
					 	 const IndexType column ) const;

#ifdef HAVE_CUDA
	__device__ __host__
#endif
	RealType getElementFast( const IndexType row,
							 const IndexType column ) const;

   // TODO: Change this to return MatrixRow type like in CSR format
	void getRow( const IndexType row,
			 	    IndexType* columns,
			 	    RealType* values ) const;

#ifdef HAVE_CUDA
	__device__ __host__
#endif
	IndexType getGroupLength( const IndexType strip,
							  const IndexType group ) const;

	template< typename InVector,
			  typename OutVector >
	void vectorProduct( const InVector& inVector,
						OutVector& outVector ) const;

	template< typename InVector,
			  typename OutVector >
	void vectorProductHost( const InVector& inVector,
							OutVector& outVector ) const;

	void setVirtualRows(const IndexType rows);

#ifdef HAVE_CUDA
	__device__ __host__
#endif
	IndexType getNumberOfGroups( const IndexType row ) const;

	bool vectorProductTest() const;

	void reset();

	bool save( File& file ) const;

	bool load( File& file );

	bool save( const String& fileName ) const;

	bool load( const String& fileName );

	void print( std::ostream& str ) const;

	void performRowBubbleSort( Containers::Vector< Index, Device, Index >& tempRowLengths );
	void computeColumnSizes( Containers::Vector< Index, Device, Index >& tempRowLengths );

//	void verifyRowLengths( const typename BiEllpack< Real, Device, Index, StripSize >::CompressedRowLengthsVector& rowLengths );

	template< typename InVector,
			  typename OutVector >
#ifdef HAVE_CUDA
	__device__
#endif
	void spmvCuda( const InVector& inVector,
				   OutVector& outVector,
				   /*const IndexType warpStart,
				   const IndexType inWarpIdx*/
				   int globalIdx ) const;

#ifdef HAVE_CUDA
	__device__ __host__
#endif
	IndexType getStripLength( const IndexType strip ) const;

#ifdef HAVE_CUDA
	__device__
#endif
	void performRowBubbleSortCudaKernel( const typename BiEllpack< Real, Device, Index, StripSize >::CompressedRowLengthsVector& rowLengths,
										 const IndexType strip );

#ifdef HAVE_CUDA
	__device__
#endif
	void computeColumnSizesCudaKernel( const typename BiEllpack< Real, Device, Index, StripSize >::CompressedRowLengthsVector& rowLengths,
									   const IndexType numberOfStrips,
									   const IndexType strip );

#ifdef HAVE_CUDA
	__device__
#endif
	IndexType power( const IndexType number,
				     const IndexType exponent ) const;

	typedef BiEllpackDeviceDependentCode< DeviceType > DeviceDependentCode;
	friend class BiEllpackDeviceDependentCode< DeviceType >;

private:

	IndexType warpSize;

	IndexType logWarpSize;

	IndexType virtualRows;

	Containers::Vector< Index, Device, Index > rowPermArray;

	Containers::Vector< Index, Device, Index > groupPointers;

};

   } //namespace Matrices
} // namespace TNL

#include <TNL/Matrices/BiEllpack_impl.h>

