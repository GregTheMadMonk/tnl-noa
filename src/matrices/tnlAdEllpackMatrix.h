#ifndef _TNLADELLPACKMATRIX_H_
#define _TNLADELLPACKMATRIX_H_

#include <matrices/tnlSparseMatrix.h>
#include <core/vectors/tnlVector.h>

template< typename Device >
class tnlAdEllpackMatrixDeviceDependentCode;

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
class tnlAdEllpackMatrix : public tnlSparseMatrix< Real, Device, Index >
{
public:

    typedef Real RealType;
    typedef Device DeviceType;
    typedef Index IndexType;
    typedef typename tnlSparseMatrix< RealType, DeviceType, IndexType >::RowLengthsVector RowLengthsVector;
    typedef tnlAdEllpackMatrix< Real, Device, Index > ThisType;
    typedef tnlAdEllpackMatrix< Real, tnlHost, Index > HostType;
    typedef tnlAdEllpackMatrix< Real, tnlCuda, Index > CudaType;

    tnlAdEllpackMatrix();

    static tnlString getType();

    tnlString getTypeVirtual() const;

    bool setDimensions( const IndexType rows,
                        const IndexType columns );

    bool setRowLengths( const RowLengthsVector& rowLengths );

    IndexType getWarp( const IndexType row ) const;

    IndexType getInWarpOffset( const IndexType row,
                               const IndexType warp ) const;

    IndexType getRowLength( const IndexType row ) const;

    template< typename Real2, typename Device2, typename Index2 >
    bool setLike( const tnlAdEllpackMatrix< Real2, Device2, Index2 >& matrix );

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

    void getRow( const IndexType row,
                 IndexType* columns,
                 RealType* values ) const;

    template< typename InVector,
              typename OutVector >
    void vectorProduct( const InVector& inVector,
                        OutVector& outVector ) const;

    bool save( tnlFile& file ) const;

    bool load( tnlFile& file );

    bool save( const tnlString& fileName ) const;

    bool load( const tnlString& fileName );

    void print( ostream& str ) const;

    bool balanceLoad( const RealType average,
                      const RowLengthsVector& rowLengths,
                      warpList* list );

    void computeWarps( const IndexType SMs,
                       const IndexType threadsPerSM,
                       warpList* list );

    bool createArrays( warpList* list );

    void performRowTest();

    void performRowLengthsTest( const RowLengthsVector& rowLengths );

    IndexType getTotalLoad() const;

#ifdef HAVE_CUDA
    template< typename InVector,
              typename OutVector >
    __device__
    void spmvCuda( const InVector& inVector,
                   OutVector& outVector,
                   const int gridIdx ) const;
#endif


    // these arrays must be public
    tnlVector< Index, Device, Index > offset;

    tnlVector< Index, Device, Index > rowOffset;

    tnlVector< Index, Device, Index > localLoad;

    tnlVector< Index, Device, Index > reduceMap;

    typedef tnlAdEllpackMatrixDeviceDependentCode< DeviceType > DeviceDependentCode;
    friend class tnlAdEllpackMatrixDeviceDependentCode< DeviceType >;
    friend class tnlAdEllpackMatrix< RealType, tnlHost, IndexType >;
    friend class tnlAdEllpackMatrix< RealType, tnlCuda, IndexType >;

protected:

    IndexType totalLoad;

    IndexType warpSize;

};

#include <implementation/matrices/tnlAdEllpackMatrix_impl.h>

#endif /* _TNLADELLPACKMATRIX_H_ */
