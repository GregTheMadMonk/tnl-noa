/***************************************************************************
                          CSR.h  -  description
                             -------------------
    begin                : Dec 10, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/Matrices/Sparse.h>
#include <TNL/Containers/Vector.h>

namespace TNL {
namespace Matrices {
   
#ifdef HAVE_UMFPACK
    template< typename Matrix, typename Preconditioner >
    class UmfpackWrapper;
#endif

template< typename Real >
class CusparseCSR;

template< typename Device >
class CSRDeviceDependentCode;

template< typename Real, typename Device = Devices::Host, typename Index = int >
class CSR : public Sparse< Real, Device, Index >
{
private:
   // convenient template alias for controlling the selection of copy-assignment operator
   template< typename Device2 >
   using Enabler = std::enable_if< ! std::is_same< Device2, Device >::value >;

   // friend class will be needed for templated assignment operators
   template< typename Real2, typename Device2, typename Index2 >
   friend class CSR;

public:

   using RealType = Real;
   //typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename Sparse< RealType, DeviceType, IndexType >:: CompressedRowLengthsVector CompressedRowLengthsVector;
   typedef typename Sparse< RealType, DeviceType, IndexType >::ConstCompressedRowLengthsVectorView ConstCompressedRowLengthsVectorView;
   typedef CSR< Real, Device, Index > ThisType;
   typedef CSR< Real, Devices::Host, Index > HostType;
   typedef CSR< Real, Devices::Cuda, Index > CudaType;
   typedef Sparse< Real, Device, Index > BaseType;
   typedef typename BaseType::MatrixRow MatrixRow;
   
   using ConstMatrixRow = typename BaseType::ConstMatrixRow;
   //using typename BaseType::ConstMatrixRow;
   //typedef SparseRow< const RealType, const IndexType > ConstMatrixRow;


   enum SPMVCudaKernel { scalar, vector, hybrid };

   CSR();

   static String getType();

   String getTypeVirtual() const;

   static String getSerializationType();

   virtual String getSerializationTypeVirtual() const;

   void setDimensions( const IndexType rows,
                       const IndexType columns );

   void setCompressedRowLengths( ConstCompressedRowLengthsVectorView rowLengths );

   IndexType getRowLength( const IndexType row ) const;

   __cuda_callable__
   IndexType getRowLengthFast( const IndexType row ) const;
   
   IndexType getNonZeroRowLength( const IndexType row ) const;

   template< typename Real2, typename Device2, typename Index2 >
   void setLike( const CSR< Real2, Device2, Index2 >& matrix );

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
   ConstMatrixRow getRow( const IndexType rowIndex ) const;

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
   void addMatrix( const CSR< Real2, Device, Index2 >& matrix,
                   const RealType& matrixMultiplicator = 1.0,
                   const RealType& thisMatrixMultiplicator = 1.0 );

   template< typename Real2, typename Index2 >
   void getTransposition( const CSR< Real2, Device, Index2 >& matrix,
                          const RealType& matrixMultiplicator = 1.0 );

   template< typename Vector1, typename Vector2 >
   bool performSORIteration( const Vector1& b,
                             const IndexType row,
                             Vector2& x,
                             const RealType& omega = 1.0 ) const;

   // copy assignment
   CSR& operator=( const CSR& matrix );

   // cross-device copy assignment
   template< typename Real2, typename Device2, typename Index2,
             typename = typename Enabler< Device2 >::type >
   CSR& operator=( const CSR< Real2, Device2, Index2 >& matrix );

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

   // The following getters allow us to interface TNL with external C-like
   // libraries such as UMFPACK or SuperLU, which need the raw data.
   const Containers::Vector< Index, Device, Index >&
   getRowPointers() const
   {
      return this->rowPointers;
   }

   Containers::Vector< Index, Device, Index >&
   getRowPointers()
   {
      return this->rowPointers;
   }

   const Containers::Vector< Index, Device, Index >&
   getColumnIndexes() const
   {
      return this->columnIndexes;
   }

   Containers::Vector< Index, Device, Index >&
   getColumnIndexes()
   {
      return this->columnIndexes;
   }

protected:

   Containers::Vector< Index, Device, Index > rowPointers;

   SPMVCudaKernel spmvCudaKernel;

   int cudaWarpSize, hybridModeSplit;

   typedef CSRDeviceDependentCode< DeviceType > DeviceDependentCode;
   
   friend class CSRDeviceDependentCode< DeviceType >;
   friend class CusparseCSR< RealType >;
};

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/CSR_impl.h>
