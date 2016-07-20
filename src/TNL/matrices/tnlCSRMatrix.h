/***************************************************************************
                          tnlCSRMatrix.h  -  description
                             -------------------
    begin                : Dec 10, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/matrices/tnlSparseMatrix.h>
#include <TNL/Vectors/Vector.h>

namespace TNL {

#ifdef HAVE_UMFPACK
    template< typename Matrix, typename Preconditioner >
    class tnlUmfpackWrapper;
#endif

template< typename Real >
class tnlCusparseCSRMatrix;

template< typename Device >
class tnlCSRMatrixDeviceDependentCode;

template< typename Real, typename Device = tnlHost, typename Index = int >
class tnlCSRMatrix : public tnlSparseMatrix< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename tnlSparseMatrix< RealType, DeviceType, IndexType >:: CompressedRowsLengthsVector CompressedRowsLengthsVector;
   typedef tnlCSRMatrix< Real, Device, Index > ThisType;
   typedef tnlCSRMatrix< Real, tnlHost, Index > HostType;
   typedef tnlCSRMatrix< Real, tnlCuda, Index > CudaType;
   typedef tnlSparseMatrix< Real, Device, Index > BaseType;
   typedef typename BaseType::MatrixRow MatrixRow;


   enum SPMVCudaKernel { scalar, vector, hybrid };

   tnlCSRMatrix();

   static String getType();

   String getTypeVirtual() const;

   bool setDimensions( const IndexType rows,
                       const IndexType columns );

   bool setCompressedRowsLengths( const CompressedRowsLengthsVector& rowLengths );

   IndexType getRowLength( const IndexType row ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const tnlCSRMatrix< Real2, Device2, Index2 >& matrix );

   void reset();

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
   // TODO: add const RealType& multiplicator = 1.0 )

   template< typename Real2, typename Index2 >
   void addMatrix( const tnlCSRMatrix< Real2, Device, Index2 >& matrix,
                   const RealType& matrixMultiplicator = 1.0,
                   const RealType& thisMatrixMultiplicator = 1.0 );

   template< typename Real2, typename Index2 >
   void getTransposition( const tnlCSRMatrix< Real2, Device, Index2 >& matrix,
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

   void setCudaKernelType( const SPMVCudaKernel kernel );

   __cuda_callable__
   SPMVCudaKernel getCudaKernelType() const;

   void setCudaWarpSize( const int warpSize );

   int getCudaWarpSize() const;

   void setHybridModeSplit( const IndexType hybridModeSplit );

   __cuda_callable__
   IndexType getHybridModeSplit() const;

#ifdef HAVE_CUDA

   template< typename InVector,
             typename OutVector,
             int warpSize >
   __device__
   void spmvCudaVectorized( const InVector& inVector,
                            OutVector& outVector,
                            const IndexType warpStart,
                            const IndexType warpEnd,
                            const IndexType inWarpIdx ) const;

   template< typename InVector,
             typename OutVector,
             int warpSize >
   __device__
   void vectorProductCuda( const InVector& inVector,
                           OutVector& outVector,
                           int gridIdx ) const;
#endif

   protected:

   Vectors::tnlVector< Index, Device, Index > rowPointers;

   SPMVCudaKernel spmvCudaKernel;

   int cudaWarpSize, hybridModeSplit;

   typedef tnlCSRMatrixDeviceDependentCode< DeviceType > DeviceDependentCode;
   friend class tnlCSRMatrixDeviceDependentCode< DeviceType >;
   friend class tnlCusparseCSRMatrix< RealType >;
#ifdef HAVE_UMFPACK
    template< typename Matrix, typename Preconditioner >
    friend class tnlUmfpackWrapper;
#endif

};

} // namespace TNL

#include <TNL/matrices/tnlCSRMatrix_impl.h>

