/***************************************************************************
                          BiEllpack.h  -  description
                             -------------------
    begin                : Aug 27, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


#pragma once

#include <matrices/SparseMatrix.h>
#include <core/vectors/Containers::Vectors::Vector.h>

namspace TNL {
   namepsace Matrices


template< typename Device >
class BiEllpackMatrixDeviceDependentCode;

template< typename Real, typename Device = Devices::Cuda, typename Index = int, int StripSize = 32 >
class BiEllpackMatrix : public SparseMatrix< Real, Device, Index >
{
public:
	typedef Real RealType;
	typedef Device DeviceType;
	typedef Index IndexType;
	typedef typename SparseMatrix< RealType, DeviceType, IndexType >::RowLengthsVector RowLengthsVector;
	typedef typename SparseMatrix< RealType, DeviceType, IndexType >::ValuesVector ValuesVector;
	typedef typename SparseMatrix< RealType, DeviceType, IndexType >::ColumnIndexesVector ColumnIndexesVector;
	typedef BiEllpackMatrix< Real, Device, Index > ThisType;
	typedef BiEllpackMatrix< Real, Devices::Host, Index > HostType;
	typedef BiEllpackMatrix< Real, Devices::Cuda, Index > CudaType;

	BiEllpackMatrix();

	static String getType();

	String getTypeVirtual() const;

	bool setDimensions( const IndexType rows,
						const IndexType columns );

	bool setRowLengths( const RowLengthsVector& rowLengths );

	IndexType getRowLength( const IndexType row ) const;

	template< typename Real2,
			  typename Device2,
			  typename Index2 >
	bool setLike( const BiEllpackMatrix< Real2, Device2, Index2, StripSize >& matrix );

	void getRowLengths( Containers::Vectors::Vector< IndexType, DeviceType, IndexType >& rowLengths ) const;

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

	bool save( tnlFile& file ) const;

	bool load( tnlFile& file );

	bool save( const String& fileName ) const;

	bool load( const String& fileName );

	void print( ostream& str ) const;

	void performRowBubbleSort( Containers::Vectors::Vector< Index, Device, Index >& tempRowLengths );
	void computeColumnSizes( Containers::Vectors::Vector< Index, Device, Index >& tempRowLengths );

//	void verifyRowLengths( const typename BiEllpackMatrix< Real, Device, Index, StripSize >::RowLengthsVector& rowLengths );

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
	void performRowBubbleSortCudaKernel( const typename BiEllpackMatrix< Real, Device, Index, StripSize >::RowLengthsVector& rowLengths,
										 const IndexType strip );

#ifdef HAVE_CUDA
	__device__
#endif
	void computeColumnSizesCudaKernel( const typename BiEllpackMatrix< Real, Device, Index, StripSize >::RowLengthsVector& rowLengths,
									   const IndexType numberOfStrips,
									   const IndexType strip );

#ifdef HAVE_CUDA
	__device__
#endif
	IndexType power( const IndexType number,
				     const IndexType exponent ) const;

	typedef BiEllpackMatrixDeviceDependentCode< DeviceType > DeviceDependentCode;
	friend class BiEllpackMatrixDeviceDependentCode< DeviceType >;

private:

	IndexType warpSize;

	IndexType logWarpSize;

	IndexType virtualRows;

	Containers::Vectors::Vector< Index, Device, Index > rowPermArray;

	Containers::Vectors::Vector< Index, Device, Index > groupPointers;

};

   } //namespace Matrices
} // namespace TNL

#include <TNL/Matrices/BiEllpackMatrix_impl.h>

