/***************************************************************************
                          BiEllpackSymmetric.h  -  description
                             -------------------
    begin                : Aug 30, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/Sparse.h>
#include <TNL/Containers/Vector.h>

namespace TNL {
namespace Matrices {
   namespace Legacy {

template< typename Device >
class BiEllpackSymmetricDeviceDependentCode;

template< typename Real, typename Device = Devices::Cuda, typename Index = int, int StripSize = 32 >
class BiEllpackSymmetric : public Sparse< Real, Device, Index >
{
public:
	typedef Real RealType;
	typedef Device DeviceType;
	typedef Index IndexType;
	typedef typename Sparse< RealType, DeviceType, IndexType >::CompressedRowLengthsVector CompressedRowLengthsVector;
   typedef typename Sparse< RealType, DeviceType, IndexType >::ConstCompressedRowLengthsVectorView ConstCompressedRowLengthsVectorView;
	typedef typename Sparse< RealType, DeviceType, IndexType >::ValuesVector ValuesVector;
	typedef typename Sparse< RealType, DeviceType, IndexType >::ColumnIndexesVector ColumnIndexesVector;

   template< typename _Real = Real,
             typename _Device = Device,
             typename _Index = Index >
   using Self = BiEllpackSymmetric< _Real, _Device, _Index >;

	BiEllpackSymmetric();

	void setDimensions( const IndexType rows, const IndexType columns );

   void setCompressedRowLengths( ConstCompressedRowLengthsVectorView rowLengths );

	IndexType getRowLength( const IndexType row ) const;

	template< typename Real2,
			  typename Device2,
			  typename Index2 >
	bool setLike( const BiEllpackSymmetric< Real2, Device2, Index2, StripSize >& matrix );

	void getRowLengths( Containers::Vector< IndexType, DeviceType, IndexType >& rowLengths ) const;

	bool setElement( const IndexType row,
					 const IndexType column,
					 const RealType& value );

   __cuda_callable__
	bool setElementFast( const IndexType row,
						 const IndexType column,
						 const RealType& value );

	bool addElement( const IndexType row,
					 const IndexType column,
					 const RealType& value,
					 const RealType& thisElementMultiplicator = 1.0 );

   __cuda_callable__
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

   __cuda_callable__
	RealType getElementFast( const IndexType row,
							 const IndexType column ) const;

	void getRow( const IndexType row,
			 	 IndexType* columns,
			 	 RealType* values ) const;

   __cuda_callable__
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

   __cuda_callable__
	IndexType getNumberOfGroups( const IndexType row ) const;

	bool vectorProductTest() const;

	void reset();

	void save( File& file ) const;

	void load( File& file );

	void save( const String& fileName ) const;

	void load( const String& fileName );

	void print( std::ostream& str ) const;

	void performRowBubbleSort( Containers::Vector< Index, Device, Index >& tempRowLengths );
	void computeColumnSizes( Containers::Vector< Index, Device, Index >& tempRowLengths );

//	void verifyRowLengths( const typename BiEllpackSymmetric< Real, Device, Index, StripSize >::RowLengthsVector& rowLengths );

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

   __cuda_callable__
	IndexType getStripLength( const IndexType strip ) const;

   __cuda_callable__
	void performRowBubbleSortCudaKernel( const typename BiEllpackSymmetric< Real, Device, Index, StripSize >::CompressedRowLengthsVector& rowLengths,
										 const IndexType strip );

   __cuda_callable__
	void computeColumnSizesCudaKernel( const typename BiEllpackSymmetric< Real, Device, Index, StripSize >::CompressedRowLengthsVector& rowLengths,
									   const IndexType numberOfStrips,
									   const IndexType strip );

   __cuda_callable__
	IndexType power( const IndexType number,
				     const IndexType exponent ) const;

	typedef BiEllpackSymmetricDeviceDependentCode< DeviceType > DeviceDependentCode;
	friend class BiEllpackSymmetricDeviceDependentCode< DeviceType >;

private:

	IndexType warpSize;

	IndexType logWarpSize;

	IndexType virtualRows;

	Containers::Vector< Index, Device, Index > rowPermArray;

	Containers::Vector< Index, Device, Index > groupPointers;

};

} //namespace Legacy
} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/BiEllpackSymmetric_impl.h>

