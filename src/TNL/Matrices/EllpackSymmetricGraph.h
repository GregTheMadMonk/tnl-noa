/***************************************************************************
                          EllpackSymmetricGraph.h  -  description
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
class EllpackSymmetricGraphDeviceDependentCode;

template< typename Real, typename Device = Devices::Host, typename Index = int >
class EllpackSymmetricGraph : public Sparse< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename Sparse< RealType, DeviceType, IndexType >::CompressedRowLengthsVector CompressedRowLengthsVector;
   typedef typename Sparse< RealType, DeviceType, IndexType >::ConstCompressedRowLengthsVectorView ConstCompressedRowLengthsVectorView;
   typedef typename Sparse< RealType, DeviceType, IndexType >::ValuesVector ValuesVector;
   typedef typename Sparse< RealType, DeviceType, IndexType >::ColumnIndexesVector ColumnIndexesVector;
   typedef EllpackSymmetricGraph< Real, Devices::Host, Index > HostType;
   typedef EllpackSymmetricGraph< Real, Devices::Cuda, Index > CudaType;


   EllpackSymmetricGraph();

   void setDimensions( const IndexType rows,
                       const IndexType columns );

   void setCompressedRowLengths( ConstCompressedRowLengthsVectorView rowLengths );

   bool setConstantRowLengths( const IndexType& rowLengths );

   IndexType getRowLength( const IndexType row ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const EllpackSymmetricGraph< Real2, Device2, Index2 >& matrix );

   void reset();

   //template< typename Real2, typename Device2, typename Index2 >
   //bool operator == ( const EllpackSymmetricGraph< Real2, Device2, Index2 >& matrix ) const;

   //template< typename Real2, typename Device2, typename Index2 >
   //bool operator != ( const EllpackSymmetricGraph< Real2, Device2, Index2 >& matrix ) const;

   /*template< typename Matrix >
   bool copyFrom( const Matrix& matrix,
                  const CompressedRowLengthsVector& rowLengths );*/

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

   template< typename InVector,
             typename OutVector >
   void vectorProductHost( const InVector& inVector,
                           OutVector& outVector ) const;

#ifdef HAVE_CUDA
   template< typename InVector,
             typename OutVector >
   __cuda_callable__
   void spmvCuda( const InVector& inVector,
                  OutVector& outVector,
                  const int globalIdx,
                  const int color ) const;
#endif

   void computePermutationArray();

   bool rearrangeMatrix( bool verbose );

   void save( File& file ) const;

   void load( File& file );

   void save( const String& fileName ) const;

   void load( const String& fileName );

   void print( std::ostream& str ) const;

   bool help( bool verbose = false );

   void verifyPermutationArray();

   __cuda_callable__
   Index getRowLengthsInt() const;

   __cuda_callable__
   Index getAlignedRows() const;

   __cuda_callable__
   Index getRowsOfColor( IndexType color ) const;

   void copyFromHostToCuda( EllpackSymmetricGraph< Real, Devices::Host, Index >& matrix );

   __cuda_callable__
   Containers::Vector< Index, Device, Index >& getPermutationArray();

   __cuda_callable__
   Containers::Vector< Index, Device, Index >& getInversePermutation();

   __cuda_callable__
   Containers::Vector< Index, Device, Index >& getColorPointers();

   protected:

   void allocateElements();

   IndexType rowLengths, alignedRows;

   typedef EllpackSymmetricGraphDeviceDependentCode< DeviceType > DeviceDependentCode;
   friend class EllpackSymmetricGraphDeviceDependentCode< DeviceType >;

   Containers::Vector< Index, Device, Index > permutationArray;
   Containers::Vector< Index, Device, Index > inversePermutationArray;
   Containers::Vector< Index, Device, Index > colorPointers;
   bool rearranged;
};

} // namespace Matrices
} // namespace TNL


#include <TNL/Matrices/EllpackSymmetricGraph_impl.h>
