/***************************************************************************
                          CSR.h  -  description
                             -------------------
    begin                : Dec 10, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/Legacy/Sparse.h>
#include <TNL/Containers/Vector.h>

#include <TNL/Devices/Cuda.h>
#include <TNL/Exceptions/CudaBadAlloc.h>
#include <vector> // vector for blocks

namespace TNL {
namespace Matrices {
   namespace Legacy {

enum class Type {
   /* LONG = 0!!! Non zero value rewrites index[1] */
   LONG = 0,
   STREAM = 1,
   VECTOR = 2
};

union Block {
   Block(uint32_t row, Type type = Type::VECTOR, uint32_t index = 0) noexcept {
      this->index[0] = row;
      this->index[1] = index;
      this->byte[7] = (uint8_t)type;
   }

   unsigned index[2]; // index[0] is row pointer, index[1] is index in warp
   uint8_t byte[8]; // byte[7] is type specificator
};

#ifdef HAVE_UMFPACK
    template< typename Matrix, typename Preconditioner >
    class UmfpackWrapper;
#endif

template< typename Real >
class CusparseCSR;

template< typename Device >
class CSRDeviceDependentCode;

enum CSRKernel { CSRScalar, CSRVector, CSRHybrid, CSRLight,
                 CSRAdaptive, CSRMultiVector, CSRLightWithoutAtomic };

template< typename Real, typename Device = Devices::Host, typename Index = int, CSRKernel KernelType = CSRScalar >
class CSR : public Sparse< Real, Device, Index >
{
private:
   // convenient template alias for controlling the selection of copy-assignment operator
   template< typename Device2 >
   using Enabler = std::enable_if< ! std::is_same< Device2, Device >::value >;

   // friend class will be needed for templated assignment operators
   template< typename Real2, typename Device2, typename Index2, CSRKernel KernelType2 >
   friend class CSR;

public:

   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   typedef typename Sparse< RealType, DeviceType, IndexType >::CompressedRowLengthsVector CompressedRowLengthsVector;
   typedef typename Sparse< RealType, DeviceType, IndexType >::CompressedRowLengthsVectorView CompressedRowLengthsVectorView;
   typedef typename Sparse< RealType, DeviceType, IndexType >::ConstCompressedRowLengthsVectorView ConstCompressedRowLengthsVectorView;
   typedef Sparse< Real, Device, Index > BaseType;
   using MatrixRow = typename BaseType::MatrixRow;
   using ConstMatrixRow = typename BaseType::ConstMatrixRow;

   template< typename _Real = Real,
             typename _Device = Device,
             typename _Index = Index >
   using Self = CSR< _Real, _Device, _Index >;

   constexpr CSRKernel getSpMVKernelType() { return KernelType; };
   //enum SPMVCudaKernel { scalar, vector, hybrid };


   Containers::Vector< Block, Device, Index > blocks;
   
   Index maxElementsPerWarp = 1024;

   using Sparse< Real, Device, Index >::getAllocatedElementsCount;

   CSR();

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

   __cuda_callable__
   IndexType getNonZeroRowLengthFast( const IndexType row ) const;

   template< typename Real2, typename Device2, typename Index2, CSRKernel KernelType2 >
   void setLike( const CSR< Real2, Device2, Index2, KernelType2 >& matrix );

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

   template< typename Real2, typename Index2, CSRKernel KernelType2 >
   void addMatrix( const CSR< Real2, Device, Index2, KernelType2 >& matrix,
                   const RealType& matrixMultiplicator = 1.0,
                   const RealType& thisMatrixMultiplicator = 1.0 );

   template< typename Real2, typename Index2, CSRKernel KernelType2 >
   void getTransposition( const CSR< Real2, Device, Index2, KernelType2 >& matrix,
                          const RealType& matrixMultiplicator = 1.0 );

   template< typename Vector1, typename Vector2 >
   bool performSORIteration( const Vector1& b,
                             const IndexType row,
                             Vector2& x,
                             const RealType& omega = 1.0 ) const;

   // copy assignment
   CSR& operator=( const CSR& matrix );

   // cross-device copy assignment
   template< typename Real2, typename Device2, typename Index2, CSRKernel KernelType2,
             typename = typename Enabler< Device2 >::type >
   CSR& operator=( const CSR< Real2, Device2, Index2, KernelType2 >& matrix );

   void save( File& file ) const;

   void load( File& file );

   void save( const String& fileName ) const;

   void load( const String& fileName );

   void print( std::ostream& str ) const;

   //void setCudaKernelType( const SPMVCudaKernel kernel );

   //__cuda_callable__
   //SPMVCudaKernel getCudaKernelType() const;

   void setCudaWarpSize( const int warpSize );

   int getCudaWarpSize() const;

   void setHybridModeSplit( const IndexType hybridModeSplit );

   __cuda_callable__
   IndexType getHybridModeSplit() const;

   /* Analyze rowPointers, columnIndecies and values to create block for CSR Adaptive */
   void setBlocks();

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

   //SPMVCudaKernel spmvCudaKernel;

   int cudaWarpSize, hybridModeSplit;

   typedef CSRDeviceDependentCode< DeviceType > DeviceDependentCode;

   friend class CSRDeviceDependentCode< DeviceType >;
   friend class CusparseCSR< RealType >;
};

} //namespace Legacy
} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/Legacy/CSR_impl.h>
