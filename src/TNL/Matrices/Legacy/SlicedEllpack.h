/***************************************************************************
                          SlicedEllpack.h  -  description
                             -------------------
    begin                : Dec 8, 2013
    copyright            : (C) 2013 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Vacata Jan
 *
 * The algorithm/method was published in:
 *  Oberhuber T., Suzuki A., Vacata J., New Row-grouped CSR format for storing
 *  the sparse matrices on GPU with implementation in CUDA, Acta Technica, 2011,
 *  vol. 56, no. 4, pp. 447-466.
 */

#pragma once

#include <TNL/Matrices/Legacy/Sparse.h>
#include <TNL/Containers/Vector.h>

namespace TNL {
namespace Matrices {
   namespace Legacy {

template< typename Device >
class SlicedEllpackDeviceDependentCode;

template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          int SliceSize = 32 >
class SlicedEllpack;

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          int SliceSize >
__global__ void SlicedEllpack_computeMaximalRowLengthInSlices_CudaKernel( SlicedEllpack< Real, Devices::Cuda, Index, SliceSize >* matrix,
                                                                          typename SlicedEllpack< Real, Devices::Cuda, Index, SliceSize >::ConstCompressedRowLengthsVectorView rowLengths,
                                                                          int gridIdx );
#endif

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
class SlicedEllpack : public Sparse< Real, Device, Index >
{
private:
   // convenient template alias for controlling the selection of copy-assignment operator
   template< typename Device2 >
   using Enabler = std::enable_if< ! std::is_same< Device2, Device >::value >;

   // friend class will be needed for templated assignment operators
   template< typename Real2, typename Device2, typename Index2, int SliceSize2 >
   friend class SlicedEllpack;

public:
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename Sparse< RealType, DeviceType, IndexType >::CompressedRowLengthsVector CompressedRowLengthsVector;
   typedef typename Sparse< RealType, DeviceType, IndexType >::ConstCompressedRowLengthsVectorView ConstCompressedRowLengthsVectorView;
   typedef typename Sparse< RealType, DeviceType, IndexType >::CompressedRowLengthsVectorView CompressedRowLengthsVectorView;
   typedef typename Sparse< RealType, DeviceType, IndexType >::ValuesVector ValuesVector;
   typedef typename Sparse< RealType, DeviceType, IndexType >::ColumnIndexesVector ColumnIndexesVector;
   typedef Sparse< Real, Device, Index > BaseType;
   typedef typename BaseType::MatrixRow MatrixRow;
   typedef SparseRow< const RealType, const IndexType > ConstMatrixRow;

   template< typename _Real = Real,
             typename _Device = Device,
             typename _Index = Index,
             int _SliceSize = SliceSize >
   using Self = SlicedEllpack< _Real, _Device, _Index, _SliceSize >;

   SlicedEllpack();

   static String getSerializationType();

   virtual String getSerializationTypeVirtual() const;

   void setDimensions( const IndexType rows,
                       const IndexType columns );

   void setCompressedRowLengths( ConstCompressedRowLengthsVectorView rowLengths );

   void getCompressedRowLengths( CompressedRowLengthsVectorView rowLengths ) const;

   IndexType getRowLength( const IndexType row ) const;

   __cuda_callable__
   IndexType getRowLengthFast( const IndexType row ) const;

   IndexType getNonZeroRowLength( const IndexType row ) const;

   template< typename Real2, typename Device2, typename Index2 >
   void setLike( const SlicedEllpack< Real2, Device2, Index2, SliceSize >& matrix );

   void reset();

   template< typename Real2, typename Device2, typename Index2 >
   bool operator == ( const SlicedEllpack< Real2, Device2, Index2 >& matrix ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool operator != ( const SlicedEllpack< Real2, Device2, Index2 >& matrix ) const;

   __cuda_callable__
   bool setElementFast( const IndexType row,
                        const IndexType column,
                        const RealType& value );

   bool setElement( const IndexType row,
                    const IndexType column,
                    const RealType& value );

   __cuda_callable__
   bool addElementFast( const IndexType row,
                        const IndexType column,
                        const RealType& value,
                        const RealType& thisElementMultiplicator = 1.0 );

   bool addElement( const IndexType row,
                    const IndexType column,
                    const RealType& value,
                    const RealType& thisElementMultiplicator = 1.0 );

   __cuda_callable__
   bool setRowFast( const IndexType row,
                    const IndexType* columnIndexes,
                    const RealType* values,
                    const IndexType elements );

   bool setRow( const IndexType row,
                const IndexType* columnIndexes,
                const RealType* values,
                const IndexType elements );

   __cuda_callable__
   bool addRowFast( const IndexType row,
                    const IndexType* columns,
                    const RealType* values,
                    const IndexType numberOfElements,
                    const RealType& thisElementMultiplicator = 1.0 );

   bool addRow( const IndexType row,
                const IndexType* columns,
                const RealType* values,
                const IndexType numberOfElements,
                const RealType& thisElementMultiplicator = 1.0 );

   __cuda_callable__
   RealType getElementFast( const IndexType row,
                            const IndexType column ) const;

   RealType getElement( const IndexType row,
                        const IndexType column ) const;

   __cuda_callable__
   void getRowFast( const IndexType row,
                    IndexType* columns,
                    RealType* values ) const;

   __cuda_callable__
   MatrixRow getRow( const IndexType rowIndex );

   __cuda_callable__
   ConstMatrixRow getRow( const IndexType rowIndex ) const;

   template< typename Vector >
   __cuda_callable__
   typename Vector::RealType rowVectorProduct( const IndexType row,
                                               const Vector& vector ) const;

   template< typename InVector,
             typename OutVector >
   void vectorProduct( const InVector& inVector,
                       OutVector& outVector,
                       RealType multiplicator = 1.0 ) const;

   template< typename Real2, typename Index2 >
   void addMatrix( const SlicedEllpack< Real2, Device, Index2 >& matrix,
                   const RealType& matrixMultiplicator = 1.0,
                   const RealType& thisMatrixMultiplicator = 1.0 );

   template< typename Real2, typename Index2 >
   void getTransposition( const SlicedEllpack< Real2, Device, Index2 >& matrix,
                          const RealType& matrixMultiplicator = 1.0 );

   template< typename Vector1, typename Vector2 >
   bool performSORIteration( const Vector1& b,
                             const IndexType row,
                             Vector2& x,
                             const RealType& omega = 1.0 ) const;

   // copy assignment
   SlicedEllpack& operator=( const SlicedEllpack& matrix );

   // cross-device copy assignment
   template< typename Real2, typename Device2, typename Index2,
             typename = typename Enabler< Device2 >::type >
   SlicedEllpack& operator=( const SlicedEllpack< Real2, Device2, Index2, SliceSize >& matrix );

   void save( File& file ) const;

   void load( File& file );

   void save( const String& fileName ) const;

   void load( const String& fileName );

   void print( std::ostream& str ) const;

protected:

   Containers::Vector< Index, Device, Index > slicePointers, sliceCompressedRowLengths;

   typedef SlicedEllpackDeviceDependentCode< DeviceType > DeviceDependentCode;
   friend class SlicedEllpackDeviceDependentCode< DeviceType >;
#ifdef HAVE_CUDA
   /*friend __global__ void SlicedEllpack_computeMaximalRowLengthInSlices_CudaKernel< Real, Index, SliceSize >( SlicedEllpack< Real, Devices::Cuda, Index, SliceSize >* matrix,
                                                                                      const typename SlicedEllpack< Real, Devices::Cuda, Index, SliceSize >::CompressedRowLengthsVector* rowLengths,
                                                                                      int gridIdx );
    */
   // TODO: The friend declaration above does not work because of __global__ storage specifier. Therefore we declare the following method as public. Fix this, when possible.

public:
   __device__ void computeMaximalRowLengthInSlicesCuda( ConstCompressedRowLengthsVectorView rowLengths,
                                                        const IndexType sliceIdx );
#endif
};

} //namespace Legacy
} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/Legacy/SlicedEllpack_impl.h>
