/***************************************************************************
                          SlicedEllpackSymmetricGraph.h  -  description
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
class SlicedEllpackSymmetricGraphDeviceDependentCode;

template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          int SliceSize = 32 >
class SlicedEllpackSymmetricGraph;

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          int SliceSize >
__global__ void SlicedEllpackSymmetricGraph_computeMaximalRowLengthInSlices_CudaKernel( SlicedEllpackSymmetricGraph< Real, Devices::Cuda, Index, SliceSize >* matrix,
                                                                                        typename SlicedEllpackSymmetricGraph< Real, Devices::Cuda, Index, SliceSize >::ConstCompressedRowLengthsVectorView rowLengths,
                                                                                        int gridIdx );
#endif

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
class SlicedEllpackSymmetricGraph : public Sparse< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename Sparse< RealType, DeviceType, IndexType >::CompressedRowLengthsVector CompressedRowLengthsVector;
   typedef typename Sparse< RealType, DeviceType, IndexType >::ConstCompressedRowLengthsVectorView ConstCompressedRowLengthsVectorView;
   typedef typename Sparse< RealType, DeviceType, IndexType >::ValuesVector ValuesVector;
   typedef typename Sparse< RealType, DeviceType, IndexType >::ColumnIndexesVector ColumnIndexesVector;
   typedef SlicedEllpackSymmetricGraph< Real, Devices::Host, Index > HostType;
   typedef SlicedEllpackSymmetricGraph< Real, Devices::Cuda, Index > CudaType;


   SlicedEllpackSymmetricGraph();

   void setDimensions( const IndexType rows,
                       const IndexType columns );

   void setCompressedRowLengths( ConstCompressedRowLengthsVectorView rowLengths );

   IndexType getRowLength( const IndexType row ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const SlicedEllpackSymmetricGraph< Real2, Device2, Index2, SliceSize >& matrix );

   void reset();

   template< typename Real2, typename Device2, typename Index2 >
   bool operator == ( const SlicedEllpackSymmetricGraph< Real2, Device2, Index2 >& matrix ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool operator != ( const SlicedEllpackSymmetricGraph< Real2, Device2, Index2 >& matrix ) const;

   template< typename InVector,
             typename OutVector >
   void vectorProductHost( const InVector& inVector, OutVector& outVector ) const;

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

   void getRow( const IndexType row,
                IndexType* columns,
                RealType* values ) const;

   template< typename Vector >
   __cuda_callable__
   typename Vector::RealType rowVectorProduct( const IndexType row,
                                               const Vector& vector ) const;

   template< typename InVector,
             typename OutVector >
   void vectorProduct( const InVector& inVector,
                       OutVector& outVector ) const;

   template< typename Real2, typename Index2 >
   void addMatrix( const SlicedEllpackSymmetricGraph< Real2, Device, Index2 >& matrix,
                   const RealType& matrixMultiplicator = 1.0,
                   const RealType& thisMatrixMultiplicator = 1.0 );

   template< typename Real2, typename Index2 >
   void getTransposition( const SlicedEllpackSymmetricGraph< Real2, Device, Index2 >& matrix,
                          const RealType& matrixMultiplicator = 1.0 );

   template< typename Vector >
   bool performSORIteration( const Vector& b,
                             const IndexType row,
                             Vector& x,
                             const RealType& omega = 1.0 ) const;

   Index getRealRowLength( const Index row );

   Containers::Vector< Index, Device, Index > getRealRowLengths();

   void save( File& file ) const;

   void load( File& file );

   void save( const String& fileName ) const;

   void load( const String& fileName );

   void print( std::ostream& str ) const;

   bool help( bool verbose = false );

#ifdef HAVE_CUDA
    template< typename InVector,
              typename OutVector >
   __device__
   void spmvCuda( const InVector& inVector,
                  OutVector& outVector,
                  const int globalIdx,
                  const int color ) const;
#endif

    void copyFromHostToCuda( SlicedEllpackSymmetricGraph< Real, Devices::Host, Index, SliceSize >& matrix );

   bool rearrangeMatrix( bool verbose = false );

   void computePermutationArray();

   Containers::Vector< Index, Device, Index > getSlicePointers();

   Containers::Vector< Index, Device, Index > getSliceRowLengths();

   Containers::Vector< Index, Device, Index > getPermutationArray();

   Containers::Vector< Index, Device, Index > getInversePermutationArray();

   Containers::Vector< Index, Device, Index > getColorPointers();

   protected:

   Containers::Vector< Index, Device, Index > slicePointers, sliceRowLengths;

   typedef SlicedEllpackSymmetricGraphDeviceDependentCode< DeviceType > DeviceDependentCode;
   friend class SlicedEllpackSymmetricGraphDeviceDependentCode< DeviceType >;

   Containers::Vector< Index, Device, Index > permutationArray;
   Containers::Vector< Index, Device, Index > inversePermutationArray;
   Containers::Vector< Index, Device, Index > colorPointers;
   bool rearranged;
#ifdef HAVE_CUDA
   /*friend __global__ void SlicedEllpackSymmetricGraph_computeMaximalRowLengthInSlices_CudaKernel< Real, Index, SliceSize >( SlicedEllpackMatrix< Real, Devices::Cuda, Index, SliceSize >* matrix,
                                                                                      const typename SlicedEllpackSymmetricGraph< Real, Devices::Cuda, Index, SliceSize >::RowLengthsVector* rowLengths,
                                                                                      int gridIdx );
    */
   // TODO: The friend declaration above does not work because of __global__ storage specifier. Therefore we declare the following method as public. Fix this, when possible.

   public:
   __device__ void computeMaximalRowLengthInSlicesCuda( ConstCompressedRowLengthsVectorView rowLengths,
                                                        const IndexType sliceIdx );

#endif

};

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/SlicedEllpackSymmetricGraph_impl.h>

