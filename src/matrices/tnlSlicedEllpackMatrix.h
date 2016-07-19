/***************************************************************************
                          tnlSlicedEllpackMatrix.h  -  description
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

#include <matrices/tnlSparseMatrix.h>
#include <core/vectors/tnlVector.h>

namespace TNL {

template< typename Device >
class tnlSlicedEllpackMatrixDeviceDependentCode;

template< typename Real = double,
          typename Device = tnlHost,
          typename Index = int,
          int SliceSize = 32 >
class tnlSlicedEllpackMatrix;

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          int SliceSize >
__global__ void tnlSlicedEllpackMatrix_computeMaximalRowLengthInSlices_CudaKernel( tnlSlicedEllpackMatrix< Real, tnlCuda, Index, SliceSize >* matrix,
                                                                                   const typename tnlSlicedEllpackMatrix< Real, tnlCuda, Index, SliceSize >::CompressedRowsLengthsVector* rowLengths,
                                                                                   int gridIdx );
#endif

template< typename Real,
          typename Device,
          typename Index,
          int SliceSize >
class tnlSlicedEllpackMatrix : public tnlSparseMatrix< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename tnlSparseMatrix< RealType, DeviceType, IndexType >::CompressedRowsLengthsVector CompressedRowsLengthsVector;
   typedef typename tnlSparseMatrix< RealType, DeviceType, IndexType >::ValuesVector ValuesVector;
   typedef typename tnlSparseMatrix< RealType, DeviceType, IndexType >::ColumnIndexesVector ColumnIndexesVector;
   typedef tnlSlicedEllpackMatrix< Real, Device, Index > ThisType;
   typedef tnlSlicedEllpackMatrix< Real, tnlHost, Index > HostType;
   typedef tnlSlicedEllpackMatrix< Real, tnlCuda, Index > CudaType;
   typedef tnlSparseMatrix< Real, Device, Index > BaseType;
   typedef typename BaseType::MatrixRow MatrixRow;


   tnlSlicedEllpackMatrix();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   bool setDimensions( const IndexType rows,
                       const IndexType columns );

   bool setCompressedRowsLengths( const CompressedRowsLengthsVector& rowLengths );

   IndexType getRowLength( const IndexType row ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const tnlSlicedEllpackMatrix< Real2, Device2, Index2, SliceSize >& matrix );

   void reset();

   template< typename Real2, typename Device2, typename Index2 >
   bool operator == ( const tnlSlicedEllpackMatrix< Real2, Device2, Index2 >& matrix ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool operator != ( const tnlSlicedEllpackMatrix< Real2, Device2, Index2 >& matrix ) const;

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
   void addMatrix( const tnlSlicedEllpackMatrix< Real2, Device, Index2 >& matrix,
                   const RealType& matrixMultiplicator = 1.0,
                   const RealType& thisMatrixMultiplicator = 1.0 );

   template< typename Real2, typename Index2 >
   void getTransposition( const tnlSlicedEllpackMatrix< Real2, Device, Index2 >& matrix,
                          const RealType& matrixMultiplicator = 1.0 );

   template< typename Vector >
   bool performSORIteration( const Vector& b,
                             const IndexType row,
                             Vector& x,
                             const RealType& omega = 1.0 ) const;

   bool save( tnlFile& file ) const;

   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   void print( std::ostream& str ) const;

   protected:

   tnlVector< Index, Device, Index > slicePointers, sliceCompressedRowsLengths;

   typedef tnlSlicedEllpackMatrixDeviceDependentCode< DeviceType > DeviceDependentCode;
   friend class tnlSlicedEllpackMatrixDeviceDependentCode< DeviceType >;
#ifdef HAVE_CUDA
   /*friend __global__ void tnlSlicedEllpackMatrix_computeMaximalRowLengthInSlices_CudaKernel< Real, Index, SliceSize >( tnlSlicedEllpackMatrix< Real, tnlCuda, Index, SliceSize >* matrix,
                                                                                      const typename tnlSlicedEllpackMatrix< Real, tnlCuda, Index, SliceSize >::CompressedRowsLengthsVector* rowLengths,
                                                                                      int gridIdx );
    */
   // TODO: The friend declaration above does not work because of __global__ storage specifier. Therefore we declare the following method as public. Fix this, when possible.

   public:
   __device__ void computeMaximalRowLengthInSlicesCuda( const CompressedRowsLengthsVector& rowLengths,
                                                        const IndexType sliceIdx );

#endif

};

} // namespace TNL

#include <matrices/tnlSlicedEllpackMatrix_impl.h>
