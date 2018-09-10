/***************************************************************************
                          AdEllpack.h  -  description
                             -------------------
    begin                : Aug 27, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/****
 * This class implements AdELL format from:
 * 
 * Maggioni M., Berger-Wolf T., 
 * AdELL: An Adaptive Warp-Balancing ELL Format for Efficient Sparse Matrix-Vector Multiplication on GPUs,
 * In proceedings of 42nd International Conference on Parallel Processing, 2013.
 */

#pragma once

#include <TNL/Matrices/Sparse.h>
#include <TNL/Containers/Vector.h>

namespace TNL {
namespace Matrices {

template< typename Device >
class AdEllpackDeviceDependentCode;

struct warpInfo
{
    int offset;
    int rowOffset;
    int localLoad;
    int reduceMap[ 32 ];

    warpInfo* next;
    warpInfo* previous;
};

class warpList
{
public:

    warpList();

    bool addWarp( const int offset,
                  const int rowOffset,
                  const int localLoad,
                  const int* reduceMap );

    warpInfo* splitInHalf( warpInfo* warp );

    int getNumberOfWarps()
    { return this->numberOfWarps; }

    warpInfo* getNextWarp( warpInfo* warp )
    { return warp->next; }

    warpInfo* getHead()
    { return this->head; }

    warpInfo* getTail()
    { return this->tail; }

    ~warpList();

private:

    int numberOfWarps;

    warpInfo* head;
    warpInfo* tail;

};

template< typename Real, typename Device, typename Index >
class AdEllpack : public Sparse< Real, Device, Index >
{
public:

    typedef Real RealType;
    typedef Device DeviceType;
    typedef Index IndexType;
    typedef typename Sparse< RealType, DeviceType, IndexType >::CompressedRowLengthsVector CompressedRowLengthsVector;
    typedef typename Sparse< RealType, DeviceType, IndexType >::ConstCompressedRowLengthsVectorView ConstCompressedRowLengthsVectorView;
    typedef AdEllpack< Real, Device, Index > ThisType;
    typedef AdEllpack< Real, Devices::Host, Index > HostType;
    typedef AdEllpack< Real, Devices::Cuda, Index > CudaType;

    AdEllpack();

    static String getType();

    String getTypeVirtual() const;

    void setCompressedRowLengths( ConstCompressedRowLengthsVectorView rowLengths );

    IndexType getWarp( const IndexType row ) const;

    IndexType getInWarpOffset( const IndexType row,
                               const IndexType warp ) const;

    IndexType getRowLength( const IndexType row ) const;

    template< typename Real2, typename Device2, typename Index2 >
    bool setLike( const AdEllpack< Real2, Device2, Index2 >& matrix );

    void reset();

    bool setElement( const IndexType row,
                     const IndexType column,
                     const RealType& value );

    bool addElement( const IndexType row,
                     const IndexType column,
                     const RealType& value,
                     const RealType& thisElementMultiplicator = 1.0 );

    bool setRow( const IndexType row,
                 const IndexType* columnIndexes,
                 const RealType* values,
                 const IndexType elements );

    bool addRow( const IndexType row,
                 const IndexType* columnIndexes,
                 const RealType* values,
                 const IndexType elements,
                 const RealType& thisElementMultiplicator = 1.0 );

    RealType getElement( const IndexType row,
                         const IndexType column ) const;

    //MatrixRow getRow( const IndexType row );

    //const MatrixType getRow( const IndexType row ) const;

    // TODO: Change this to return MatrixRow type like in CSR format, like those above
    void getRow( const IndexType row,
                 IndexType* columns,
                 RealType* values ) const;

    template< typename InVector,
              typename OutVector >
    void vectorProduct( const InVector& inVector,
                        OutVector& outVector ) const;

    bool save( File& file ) const;

    bool load( File& file );

    bool save( const String& fileName ) const;

    bool load( const String& fileName );

    void print( std::ostream& str ) const;

    bool balanceLoad( const RealType average,
                      ConstCompressedRowLengthsVectorView rowLengths,
                      warpList* list );

    void computeWarps( const IndexType SMs,
                       const IndexType threadsPerSM,
                       warpList* list );

    bool createArrays( warpList* list );

    void performRowTest();

    void performRowLengthsTest( ConstCompressedRowLengthsVectorView rowLengths );

    IndexType getTotalLoad() const;

#ifdef HAVE_CUDA
    template< typename InVector,
              typename OutVector >
    __device__
    void spmvCuda( const InVector& inVector,
                   OutVector& outVector,
                   const int gridIdx ) const;

    template< typename InVector,
              typename OutVector >
   __device__
   void spmvCuda2( const InVector& inVector,
                   OutVector& outVector,
                   const int gridIdx ) const;

   template< typename InVector,
             typename OutVector >
   __device__
   void spmvCuda4( const InVector& inVector,
                   OutVector& outVector,
                   const int gridIdx ) const;
   
   template< typename InVector,
          typename OutVector >
   __device__
   void spmvCuda8( const InVector& inVector,
                   OutVector& outVector,
                   const int gridIdx ) const;
   
   template< typename InVector,
          typename OutVector >
   __device__
   void spmvCuda16( const InVector& inVector,
                    OutVector& outVector,
                    const int gridIdx ) const;   

   template< typename InVector,
          typename OutVector >
   __device__
   void spmvCuda32( const InVector& inVector,
                    OutVector& outVector,
                    const int gridIdx ) const;   
   
   
#endif


    // these arrays must be public
    Containers::Vector< Index, Device, Index > offset;

    Containers::Vector< Index, Device, Index > rowOffset;

    Containers::Vector< Index, Device, Index > localLoad;

    Containers::Vector< Index, Device, Index > reduceMap;

    typedef AdEllpackDeviceDependentCode< DeviceType > DeviceDependentCode;
    friend class AdEllpackDeviceDependentCode< DeviceType >;
    friend class AdEllpack< RealType, Devices::Host, IndexType >;
    friend class AdEllpack< RealType, Devices::Cuda, IndexType >;

protected:

    IndexType totalLoad;

    IndexType warpSize;

};

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/AdEllpack_impl.h>
