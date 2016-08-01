/***************************************************************************
                          SlicedEllpackMatrix.h  -  description
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

#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Vectors/Vector.h>

namespace TNL {
namespace Matrices {   

template< typename Device >
class SlicedEllpackMatrixDeviceDependentCode;

template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          int SliceSize = 32 >
class SlicedEllpackMatrix;

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          int SliceSize >
__global__ void SlicedEllpackMatrix_computeMaximalRowLengthInSlices_CudaKernel( SlicedEllpackMatrix< Real, Devices::Cuda, Index, SliceSize >* matrix,
                                                                                   const typename SlicedEllpackMatrix< Real, Devices::Cuda, Index, SliceSize >::CompressedRowsLengthsVector* rowLengths,
                                                                                   int gridIdx );
#endif

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
class SlicedEllpackMatrix : public SparseMatrix< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename SparseMatrix< RealType, DeviceType, IndexType >::CompressedRowsLengthsVector CompressedRowsLengthsVector;
   typedef typename SparseMatrix< RealType, DeviceType, IndexType >::ValuesVector ValuesVector;
   typedef typename SparseMatrix< RealType, DeviceType, IndexType >::ColumnIndexesVector ColumnIndexesVector;
   typedef SlicedEllpackMatrix< Real, Device, Index > ThisType;
   typedef SlicedEllpackMatrix< Real, Devices::Host, Index > HostType;
   typedef SlicedEllpackMatrix< Real, Devices::Cuda, Index > CudaType;
   typedef SparseMatrix< Real, Device, Index > BaseType;
   typedef typename BaseType::MatrixRow MatrixRow;


   SlicedEllpackMatrix();

   static String getType();

   String getTypeVirtual() const;

   bool setDimensions( const IndexType rows,
                       const IndexType columns );

   bool setCompressedRowsLengths( const CompressedRowsLengthsVector& rowLengths );

   IndexType getRowLength( const IndexType row ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const SlicedEllpackMatrix< Real2, Device2, Index2, SliceSize >& matrix );

   void reset();

   template< typename Real2, typename Device2, typename Index2 >
   bool operator == ( const SlicedEllpackMatrix< Real2, Device2, Index2 >& matrix ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool operator != ( const SlicedEllpackMatrix< Real2, Device2, Index2 >& matrix ) const;

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
   const MatrixRow getRow( const IndexType rowIndex ) const;

   template< typename Vector >
   __cuda_callable__
   typename Vector::RealType rowVectorProduct( const IndexType row,
                                               const Vector& vector ) const;

   template< typename InVector,
             typename OutVector >
   void vectorProduct( const InVector& inVector,
                       OutVector& outVector ) const;

   template< typename Real2, typename Index2 >
   void addMatrix( const SlicedEllpackMatrix< Real2, Device, Index2 >& matrix,
                   const RealType& matrixMultiplicator = 1.0,
                   const RealType& thisMatrixMultiplicator = 1.0 );

   template< typename Real2, typename Index2 >
   void getTransposition( const SlicedEllpackMatrix< Real2, Device, Index2 >& matrix,
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

   Vectors::Vector< Index, Device, Index > slicePointers, sliceCompressedRowsLengths;

   typedef SlicedEllpackMatrixDeviceDependentCode< DeviceType > DeviceDependentCode;
   friend class SlicedEllpackMatrixDeviceDependentCode< DeviceType >;
#ifdef HAVE_CUDA
   /*friend __global__ void SlicedEllpackMatrix_computeMaximalRowLengthInSlices_CudaKernel< Real, Index, SliceSize >( SlicedEllpackMatrix< Real, Devices::Cuda, Index, SliceSize >* matrix,
                                                                                      const typename SlicedEllpackMatrix< Real, Devices::Cuda, Index, SliceSize >::CompressedRowsLengthsVector* rowLengths,
                                                                                      int gridIdx );
    */
   // TODO: The friend declaration above does not work because of __global__ storage specifier. Therefore we declare the following method as public. Fix this, when possible.

   public:
   __device__ void computeMaximalRowLengthInSlicesCuda( const CompressedRowsLengthsVector& rowLengths,
                                                        const IndexType sliceIdx );

#endif

};

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/SlicedEllpackMatrix_impl.h>
