/***************************************************************************
                          SlocedEllpackSymmetric.h  -  description
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

template< typename Device >
class SlicedEllpackSymmetricDeviceDependentCode;

template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          int SliceSize = 32 >
class SlicedEllpackSymmetric;

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          int SliceSize >
__global__ void SlicedEllpackSymmetric_computeMaximalRowLengthInSlices_CudaKernel( SlicedEllpack< Real, Devices::Cuda, Index, SliceSize >* matrix,
                                                                                   const typename SlicedEllpackSymmetric< Real, Devices::Cuda, Index, SliceSize >::CompressedRowLengthsVector* rowLengths,
                                                                                   int gridIdx );
#endif

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
class SlicedEllpackSymmetric : public Sparse< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename Sparse< RealType, DeviceType, IndexType >::CompressedRowLengthsVector CompressedRowLengthsVector;
   typedef typename Sparse< RealType, DeviceType, IndexType >::ValuesVector ValuesVector;
   typedef typename Sparse< RealType, DeviceType, IndexType >::ColumnIndexesVector ColumnIndexesVector;
   typedef SlicedEllpackSymmetric< Real, Device, Index > ThisType;
   typedef SlicedEllpackSymmetric< Real, Devices::Host, Index > HostType;
   typedef SlicedEllpackSymmetric< Real, Devices::Cuda, Index > CudaType;


   SlicedEllpackSymmetric();

   static String getType();

   String getTypeVirtual() const;

   void setDimensions( const IndexType rows,
                       const IndexType columns );

   void setCompressedRowLengths( const CompressedRowLengthsVector& rowLengths );

   IndexType getRowLength( const IndexType row ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const SlicedEllpackSymmetric< Real2, Device2, Index2, SliceSize >& matrix );

   void reset();

   template< typename Real2, typename Device2, typename Index2 >
   bool operator == ( const SlicedEllpackSymmetric< Real2, Device2, Index2 >& matrix ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool operator != ( const SlicedEllpackSymmetric< Real2, Device2, Index2 >& matrix ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   bool setElementFast( const IndexType row,
                        const IndexType column,
                        const RealType& value );

   bool setElement( const IndexType row,
                    const IndexType column,
                    const RealType& value );

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   bool addElementFast( const IndexType row,
                        const IndexType column,
                        const RealType& value,
                        const RealType& thisElementMultiplicator = 1.0 );

   bool addElement( const IndexType row,
                    const IndexType column,
                    const RealType& value,
                    const RealType& thisElementMultiplicator = 1.0 );

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   bool setRowFast( const IndexType row,
                    const IndexType* columnIndexes,
                    const RealType* values,
                    const IndexType elements );

   bool setRow( const IndexType row,
                const IndexType* columnIndexes,
                const RealType* values,
                const IndexType elements );

#ifdef HAVE_CUDA
   __device__ __host__
#endif
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

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   RealType getElementFast( const IndexType row,
                            const IndexType column ) const;

   RealType getElement( const IndexType row,
                        const IndexType column ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   void getRowFast( const IndexType row,
                    IndexType* columns,
                    RealType* values ) const;

   void getRow( const IndexType row,
                IndexType* columns,
                RealType* values ) const;

   template< typename InVector,
             typename OutVector >
   #ifdef HAVE_CUDA
      __device__ __host__
   #endif
   void rowVectorProduct( const IndexType row,
                          const InVector& inVector,
                          OutVector& outVector ) const;

   template< typename InVector,
             typename OutVector >
   void vectorProduct( const InVector& inVector,
                       OutVector& outVector ) const;

   template< typename InVector,
             typename OutVector >
#ifdef HAVE_CUDA
   __device__
#endif
   void spmvCuda( const InVector& inVector,
                  OutVector& outVector,
                  int globalIdx ) const;

   template< typename Real2, typename Index2 >
   void addMatrix( const SlicedEllpackSymmetric< Real2, Device, Index2 >& matrix,
                   const RealType& matrixMultiplicator = 1.0,
                   const RealType& thisMatrixMultiplicator = 1.0 );

   template< typename Real2, typename Index2 >
   void getTransposition( const SlicedEllpackSymmetric< Real2, Device, Index2 >& matrix,
                          const RealType& matrixMultiplicator = 1.0 );

   template< typename Vector >
   bool performSORIteration( const Vector& b,
                             const IndexType row,
                             Vector& x,
                             const RealType& omega = 1.0 ) const;

   bool save( File& file ) const;

   bool load( File& file );

   bool save( const String& fileName ) const;

   bool load( const String& fileName );

   void print( std::ostream& str ) const;

   protected:

   Containers::Vector< Index, Device, Index > slicePointers, sliceRowLengths;

   typedef SlicedEllpackSymmetricDeviceDependentCode< DeviceType > DeviceDependentCode;
   friend class SlicedEllpackSymmetricDeviceDependentCode< DeviceType >;
#ifdef HAVE_CUDA
   /*friend __global__ void SlicedEllpackSymmetric_computeMaximalRowLengthInSlices_CudaKernel< Real, Index, SliceSize >( SlicedEllpackMatrix< Real, Devices::Cuda, Index, SliceSize >* matrix,
                                                                                      const typename SlicedEllpackSymmetric< Real, Devices::Cuda, Index, SliceSize >::RowLengthsVector* rowLengths,
                                                                                      int gridIdx );
    */
   // TODO: The friend declaration above does not work because of __global__ storage specifier. Therefore we declare the following method as public. Fix this, when possible.

   public:
   __device__ void computeMaximalRowLengthInSlicesCuda( const CompressedRowLengthsVector& rowLengths,
                                                        const IndexType sliceIdx );

#endif

};

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/SlicedEllpackSymmetric_impl.h>
