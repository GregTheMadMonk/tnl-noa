#ifndef _TNLADELLPACKMATRIX_IMPL_H_
#define _TNLADELLPACKMATRIX_IMPL_H_

#include <matrices/tnlAdEllpackMatrix.h>
#include <core/vectors/tnlVector.h>
#include <core/mfuncs.h>

/*
 * Auxiliary list implementation
 */

warpList::warpList()
{
    this->head = new warpInfo;
    this->tail = new warpInfo;
    this->head->previous = nullptr;
    this->head->next = this->tail;
    this->tail->previous = this->head;
    this->tail->next = nullptr;

    this->numberOfWarps = 0;
}

bool warpList::addWarp( const int offset,
                        const int rowOffset,
                        const int localLoad,
                        const int* reduceMap )
{
    warpInfo* temp = new warpInfo();
    if( !temp )
        return false;
    temp->offset = offset;
    temp->rowOffset = rowOffset;
    temp->localLoad = localLoad;
    for( int i = 0; i < 32; i++ )
        temp->reduceMap[ i ] = reduceMap[ i ];
    temp->next = this->tail;
    temp->previous = this->tail->previous;
    temp->previous->next = temp;
    this->tail->previous = temp;

    this->numberOfWarps++;

    return true;
}

warpInfo* warpList::splitInHalf( warpInfo* warp )
{
    warpInfo* firstHalf = new warpInfo();
    warpInfo* secondHalf = new warpInfo();
    int localLoad = ( warp->localLoad / 2 ) + ( warp->localLoad % 2 == 0 ? 0 : 1 );

    int rowOffset = warp->rowOffset;

    // first half split
    firstHalf->localLoad = localLoad;
    firstHalf->rowOffset = warp->rowOffset;
    firstHalf->offset = warp->offset;

    firstHalf->reduceMap[ 0 ] = 1;
    firstHalf->reduceMap[ 1 ] = 0;
    for( int i = 1; i < 16; i++ )
    {
        if( warp->reduceMap[ i ] == 1 )
        {
            rowOffset++;
            firstHalf->reduceMap[ 2 * i ] = 1;
            firstHalf->reduceMap[ 2 * i + 1 ] = 0;
        }
        else
        {
            firstHalf->reduceMap[ 2 * i ] = 0;
            firstHalf->reduceMap[ 2 * i + 1 ] = 0;
        }
    }

    // second half split
    secondHalf->rowOffset = rowOffset;
    if( warp->reduceMap[ 16 ] == 1 )
        secondHalf->rowOffset = rowOffset + 1;
    secondHalf->offset = 32 * firstHalf->localLoad + firstHalf->offset;
    secondHalf->reduceMap[ 0 ] = 1;
    secondHalf->reduceMap[ 1 ] = 0;
    for( int i = 1; i < 16; i++ )
    {
        if( warp->reduceMap[ i + 16 ] == 1 )
        {
            secondHalf->reduceMap[ 2 * i ] = 1;
            secondHalf->reduceMap[ 2 * i + 1 ] = 0;
        }
        else
        {
            secondHalf->reduceMap[ 2 * i ] = 0;
            secondHalf->reduceMap[ 2 * i + 1 ] = 0;
        }
    }
    secondHalf->localLoad = localLoad;

    // and warps must be connected to the list
    firstHalf->next = secondHalf;
    secondHalf->previous = firstHalf;
    firstHalf->previous = warp->previous;
    warp->previous->next = firstHalf;
    secondHalf->next = warp->next;
    warp->next->previous = secondHalf;

    // if original load was odd, all next warp offsets must be changed!
    if( ( warp->localLoad % 2 ) != 0 )
    {
        warp = secondHalf->next;
        while( warp != this->tail )
        {
            warp->offset = warp->previous->offset + warp->previous->localLoad * 32;
            warp = warp->next;
        }
    }
    this->numberOfWarps++;

    // method returns the first of the two splitted warps
    return firstHalf;
}

warpList::~warpList()
{
    while( this->head->next != nullptr )
    {
        warpInfo* temp = new warpInfo;
        temp = this->head->next;
        this->head->next = temp->next;
        delete temp;
    }
    delete this->head;
}


/*
 * Adaptive Ellpack implementation
 */

template< typename Real,
          typename Device,
          typename Index >
tnlAdEllpackMatrix< Real, Device, Index >::tnlAdEllpackMatrix()
:
warpSize( 32 )
{}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlAdEllpackMatrix< Real, Device, Index >::getTypeVirtual() const
{
    return this->getType();
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlAdEllpackMatrix< Real, Device, Index >::getType()
{
    return tnlString( "tnlAdEllpackMatrix< ") +
           tnlString( ::getType< Real >() ) +
           tnlString( ", " ) +
           Device :: getDeviceType() +
           tnlString( " >" );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlAdEllpackMatrix< Real, Device, Index >::setDimensions( const IndexType row,
                                                               const IndexType column )
{
    //TODO: implement this
    if( !tnlSparseMatrix< Real, Device, Index >::setDimensions( row, column ) )
        return false;
    return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlAdEllpackMatrix< Real, Device, Index >::setRowLengths( const RowLengthsVector& rowLengths )
{
    tnlAssert( this->getRows() > 0, );
    tnlAssert( this->getColumns() > 0, );
    if( DeviceType::DeviceType == ( int ) tnlHostDevice )
    {
        RealType average = 0.0;
        for( IndexType row = 0; row < this->getRows(); row++ )
            average += rowLengths.getElement( row );
        average /= ( RealType ) this->getRows();

        warpList* list = new warpList();

        if( !this->balanceLoad( average, rowLengths, list ) )
            return false;

        IndexType SMs = 8;
        IndexType threadsPerSM = 512;

#ifdef HAVE_CUDA
        SMs = Device::getSMs();
        threadsPerSM = Device::getThreadsPerSM();
#endif

        this->computeWarps( SMs, threadsPerSM, list );

        if( !this->createArrays( list ) )
            return false;

        //this->performRowTest();

        //cout << "========================" << endl;
        //cout << "Testing row lengths" << endl;
        //cout << "========================" << endl;
        //this->performRowLengthsTest( rowLengths );
    }

    if( DeviceType::DeviceType == ( int ) tnlCudaDevice )
    {
        tnlAdEllpackMatrix< RealType, tnlHost, IndexType > hostMatrix;
        hostMatrix.setDimensions( this->getRows(), this->getColumns() );
        tnlVector< IndexType, tnlHost, IndexType > hostRowLengths;
        hostRowLengths.setLike( rowLengths );
        hostRowLengths = rowLengths;
        hostMatrix.setRowLengths( hostRowLengths );

        this->offset.setLike( hostMatrix.offset );
        this->offset = hostMatrix.offset;
        this->rowOffset.setLike( hostMatrix.rowOffset );
        this->rowOffset = hostMatrix.rowOffset;
        this->localLoad.setLike( hostMatrix.localLoad );
        this->localLoad = hostMatrix.localLoad;
        this->reduceMap.setLike( hostMatrix.reduceMap );
        this->reduceMap = hostMatrix.reduceMap;

        this->allocateMatrixElements( this->offset.getElement( this->offset.getSize() - 1 ) );
    }
    return true;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlAdEllpackMatrix< Real, Device, Index >::performRowLengthsTest( const RowLengthsVector& rowLengths )
{
    bool found = false;
    for( IndexType row = 0; row < this->getRows(); row++ )
    {
        found = false;
        IndexType warp = this->getWarp( row );
        IndexType inWarpOffset = this->getInWarpOffset( row, warp );
        if( inWarpOffset == 0 && rowOffset.getElement( warp ) != row )
            warp++;
        IndexType rowLength = this->localLoad.getElement( warp );
        while( !found )
        {
            if( ( inWarpOffset < this->warpSize - 1 ) &&
                ( this->reduceMap.getElement( this->warpSize * warp + inWarpOffset + 1 ) == row ) )
		{
                    inWarpOffset++;
		    rowLength += this->localLoad.getElement( warp );
		}
            else if( ( inWarpOffset == this->warpSize - 1 ) &&
                     ( this->rowOffset.getElement( warp + 1 ) == row ) )
            {
                warp++;
                inWarpOffset = 0;
		rowLength += localLoad.getElement( warp );
            }
            else
                found = true;
        }
	if( rowLength < rowLengths.getElement( row ) )
	    cout << "Row: " << row << " is short!";
    }
}

template< typename Real,
          typename Device,
          typename Index >
void tnlAdEllpackMatrix< Real, Device, Index >::performRowTest()
{
    for( IndexType warp = 0; warp < this->localLoad.getSize(); warp++ )
    {
	IndexType row = this->rowOffset.getElement( warp );
        for( IndexType i = warp * this->warpSize + 1; i < ( warp + 1 ) * this->warpSize; i++ )
	{
	    if( this->reduceMap.getElement( i ) == 1 )
		row++;
	}
	if( row == this->rowOffset.getElement( warp + 1 ) || row + 1 == this->rowOffset.getElement( warp + 1 ) )
	    ;
	else 
        {
	    cout << "Error warp = " << warp << endl;
	    cout << "Row: " << row << ", Row offset: " << this->rowOffset.getElement( warp + 1 ) << endl;
        }
    }
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlAdEllpackMatrix< Real, Device, Index >::getWarp( const IndexType row ) const
{
    for( IndexType searchedWarp = 0; searchedWarp < this->getRows(); searchedWarp++ )
    {
        if( ( this->rowOffset.getElement( searchedWarp ) == row ) ||
            ( ( this->rowOffset.getElement( searchedWarp ) < row ) && ( this->rowOffset.getElement( searchedWarp + 1 ) >= row ) ) )
            return searchedWarp;
    }
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlAdEllpackMatrix< Real, Device, Index >::getInWarpOffset( const IndexType row,
                                                                  const IndexType warp ) const
{
    IndexType inWarpOffset = warp * this->warpSize;
    IndexType currentRow = this->rowOffset.getElement( warp );
    while( ( inWarpOffset < ( warp + 1 ) * this->warpSize ) && ( currentRow < row ) )
    {
        inWarpOffset++;
	if( this->reduceMap.getElement( inWarpOffset ) == 1 )
            currentRow++;
    }
    return ( inWarpOffset & ( this->warpSize - 1 ) );
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlAdEllpackMatrix< Real, Device, Index >::getRowLength( const IndexType row ) const
{
    IndexType warp = this->getWarp( row );
    IndexType inWarpOffset = this->getInWarpOffset( row, warp );
    if( inWarpOffset == 0 && rowOffset.getElement( warp ) != row )
        warp++;

    bool found = false;
    IndexType length = 0;
    IndexType elementPtr;
    while( !found )
    {
        elementPtr = this->offset.getElement( warp ) + inWarpOffset;
        for( IndexType i = 0; i < this->localLoad.getElement( warp ); i++ )
        {
            if( this->columnIndexes.getElement( elementPtr ) != this->getPaddingIndex() )
                length++;
            elementPtr += this->warpSize;
        }
        if( ( inWarpOffset < this->warpSize - 1 ) &&
            ( this->reduceMap.getElement( this->warpSize * warp + inWarpOffset + 1 ) == 0 ) )
            inWarpOffset++;
        else if( ( inWarpOffset == this->warpSize - 1 ) &&
                 ( this->rowOffset.getElement( warp + 1 ) == row ) )
        {
            warp++;
            inWarpOffset = 0;
        }
        else
            found = true;
    }
    return length;
}

template< typename Real,
          typename Device,
          typename Index >
template< typename Real2,
          typename Device2,
          typename Index2 >
bool tnlAdEllpackMatrix< Real, Device, Index >::setLike( const tnlAdEllpackMatrix< Real2, Device2, Index2 >& matrix )
{
    if( !tnlSparseMatrix< Real, Device, Index >::setLike( matrix ) ||
        !this->offset.setLike( matrix.offset ) ||
        !this->rowOffset.setLike( matrix.rowOffset ) ||
        !this->localLoad.setLike( matrix.localLoad ) ||
        !this->reduceMap.setLike( matrix.reduceMap ) )
        return false;
    return true;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlAdEllpackMatrix< Real, Device, Index >::reset()
{
    tnlSparseMatrix< Real, Device, Index >::reset();
    this->offset.reset();
    this->rowOffset.reset();
    this->localLoad.reset();
    this->reduceMap.reset();
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlAdEllpackMatrix< Real, Device, Index >::setElement( const IndexType row,
                                                            const IndexType column,
                                                            const RealType& value )
{
    return this->addElement( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlAdEllpackMatrix< Real, Device, Index >::addElement( const IndexType row,
                                                            const IndexType column,
                                                            const RealType& value,
                                                            const RealType& thisElementMultiplicator )
{
    IndexType warp = this->getWarp( row );
    IndexType inWarpOffset = this->getInWarpOffset( row, warp );
    if( inWarpOffset == 0 && rowOffset.getElement( warp ) != row )
        warp++;

    bool found = false;
    IndexType elementPtr;
    IndexType iterator = 0;
    while( !found )
    {
        elementPtr = this->offset.getElement( warp ) + inWarpOffset;
        for( IndexType i = 0; i < this->localLoad.getElement( warp ); i++ )
        {
            iterator++;
            if( this->columnIndexes.getElement( elementPtr ) == column )
            {
                this->values.setElement( elementPtr, thisElementMultiplicator * this->values.getElement( elementPtr ) + value );
                return true;
            }
            if( this->columnIndexes.getElement( elementPtr ) == this->getPaddingIndex() )
            {
                this->columnIndexes.setElement( elementPtr, column );
                this->values.setElement( elementPtr, value );
                return true;
            }
            elementPtr += this->warpSize;
        }
        if( ( inWarpOffset < this->warpSize - 1 ) &&
            ( this->reduceMap.getElement( this->warpSize * warp + inWarpOffset + 1 ) == 0 ) )
	{
            inWarpOffset++;
	}
        else if( ( inWarpOffset == this->warpSize - 1 ) &&
                 ( this->rowOffset.getElement( warp + 1 ) == row ) )
        {
            warp++;
            inWarpOffset = 0;
        }
        else
            found = true;
    }
    return false;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlAdEllpackMatrix< Real, Device, Index >::setRow( const IndexType row,
                                                        const IndexType* columnIndexes,
                                                        const RealType* values,
                                                        const IndexType elements )
{
    IndexType warp = this->getWarp( row );
    IndexType inWarpOffset = this->getInWarpOffset( row, warp );
    if( inWarpOffset == 0 && rowOffset.getElement( warp ) != row )
        warp++;

    bool found = false;
    IndexType length = 0;
    IndexType elementPtr;
    IndexType elPtr = 0;
    while( ( !found ) && ( elPtr < elements ) )
    {
        elementPtr = this->offset.getElement( warp ) + inWarpOffset;
        for( IndexType i = 0; ( i < this->localLoad.getElement( warp ) ) && ( elPtr < elements ); i++ )
        {
            this->values.setElement( elementPtr, values[ elPtr ] );
            this->columnIndexes.setElement( elementPtr, columnIndexes[ elPtr ] );
            elPtr++;
            elementPtr += this->warpSize;
        }
        if( ( inWarpOffset < this->warpSize - 1 ) &&
            ( this->reduceMap.getElement( this->warpSize * warp + inWarpOffset + 1 ) == 0 ) )
            inWarpOffset++;
        else if( ( inWarpOffset == this->warpSize - 1 ) &&
                 ( this->rowOffset.getElement( warp + 1 ) == row ) )
        {
            warp++;
            inWarpOffset = 0;
        }
        else
            found = true;
    }
    return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlAdEllpackMatrix< Real, Device, Index >::addRow( const IndexType row,
                                                        const IndexType* columnIndexes,
                                                        const RealType* values,
                                                        const IndexType elements,
                                                        const RealType& thisElementMultiplicator )
{
    return false;
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlAdEllpackMatrix< Real, Device, Index >::getElement( const IndexType row,
                                                            const IndexType column ) const
{
    IndexType warp = this->getWarp( row );
    IndexType inWarpOffset = this->getInWarpOffset( row, warp );
    if( inWarpOffset == 0 && rowOffset.getElement( warp ) != row )
        warp++;

    bool found = false;
    IndexType elementPtr;
    while( !found )
    {
        elementPtr = this->offset.getElement( warp ) + inWarpOffset;
        for( IndexType i = 0; i < this->localLoad.getElement( warp ); i++ )
        {
            if( this->columnIndexes.getElement( elementPtr ) == column )
                return this->values.getElement( elementPtr );
            elementPtr += this->warpSize;
        }
        if( ( inWarpOffset < this->warpSize - 1 ) &&
            ( this->reduceMap.getElement( this->warpSize * warp + inWarpOffset + 1 ) == 0 ) )
            inWarpOffset++;
        else if( ( inWarpOffset == this->warpSize - 1 ) &&
                 ( this->rowOffset.getElement( warp + 1 ) == row ) )
        {
            warp++;
            inWarpOffset = 0;
        }
        else
            found = true;
    }
    return 0.0;
}
/*
template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
    __device__ __host__
#endif
typename tnlAdEllpackMatrix< Real, Device, Index >::MatrixRow
tnlAdEllpackMatrix< Real, Device, Index >::getRow( const IndexType row )
{

}*/

template< typename Real,
          typename Device,
          typename Index >
void tnlAdEllpackMatrix< Real, Device, Index >::getRow( const IndexType row,
                                                        IndexType* columns,
                                                        RealType* values ) const
{
	;
}
/*
template< typename Real,
          typename Device,
          typename Index >
#ifdef HAVE_CUDA
    __device__ __host__
#endif
const typedef tnlAdEllpackMatrix< Real, Device, Index >::MatrixRow
tnlAdEllpackMatrix< Real, Device, Index >::getRow( const IndexType row ) const
{

}*/

template< typename Real,
          typename Device,
          typename Index >
template< typename InVector,
          typename OutVector >
void tnlAdEllpackMatrix< Real, Device, Index >::vectorProduct( const InVector& inVector,
                                                               OutVector& outVector ) const
{
    DeviceDependentCode::vectorProduct( *this, inVector, outVector );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlAdEllpackMatrix< Real, Device, Index >::save( tnlFile& file ) const
{
    if( tnlSparseMatrix< Real, Device, Index >::save( file ) ||
        this->offset.save( file ) ||
        this->rowOffset.save( file ) ||
        this->localLoad.save( file ) ||
        this->reduceMap.save( file ) )
        return false;
    return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlAdEllpackMatrix< Real, Device, Index >::load( tnlFile& file )
{
    if( tnlSparseMatrix< Real, Device, Index >::load( file ) ||
        this->offset.load( file ) ||
        this->rowOffset.load( file ) ||
        this->localLoad.load( file ) ||
        this->reduceMap.load( file ) )
        return false;
    return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlAdEllpackMatrix< Real, Device, Index >::save( const tnlString& fileName ) const
{
    return tnlObject::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlAdEllpackMatrix< Real, Device, Index >::load( const tnlString& fileName )
{
    return tnlObject::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
void tnlAdEllpackMatrix< Real, Device, Index >::print( ostream& str ) const
{
    for( IndexType row = 0; row < this->getRows(); row++ )
    {
        str  << "Row: " << row << " -> \t";

        IndexType warp = this->getWarp( row );
        IndexType inWarpOffset = this->getInWarpOffset( row, warp );
        if( inWarpOffset == 0 && rowOffset.getElement( warp ) != row )
            warp++;

        bool found = false;
        IndexType elementPtr;
        while( !found )
        {
            elementPtr = this->offset.getElement( warp ) + inWarpOffset;
            for( IndexType i = 0; i < this->localLoad.getElement( warp ); i++ )
            {
                if( this->columnIndexes.getElement( elementPtr ) != this->getPaddingIndex() )
                    str << " column: " << this->columnIndexes.getElement( elementPtr ) << " -> "
                        << " value: " << this->values.getElement( elementPtr ) << endl;
                elementPtr += this->warpSize;
            }
            if( ( inWarpOffset < this->warpSize - 1 ) &&
                ( this->reduceMap.getElement( this->warpSize * warp + inWarpOffset + 1 ) == 0 ) )
                inWarpOffset++;
            else if( ( inWarpOffset == this->warpSize - 1 ) &&
                     ( this->rowOffset.getElement( warp + 1 ) == row ) )
            {
                warp++;
                inWarpOffset = 0;
            }
            else
                found = true;
        }
    }
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlAdEllpackMatrix< Real, Device, Index >::balanceLoad( const RealType average,
                                                             const RowLengthsVector& rowLengths,
                                                             warpList* list )
{
    IndexType offset, rowOffset, localLoad, reduceMap[ 32 ];

    IndexType numberOfThreads = 0;
    IndexType ave = average * 1.1;
    offset = 0;
    rowOffset = 0;
    bool addedWarp = false;
    IndexType warpId = 0;

    for( IndexType row = 0; row < this->getRows(); row++ )
    {
        addedWarp = false;
        // takes every row
        IndexType rowLength = rowLengths.getElement( row );
        // counts necessary number of threads
        IndexType threadsPerRow = ( IndexType ) ( rowLength / ave ) + ( rowLength % ave == 0 ? 0 : 1 );
        // if this number added to number of total threads in this warp is higher than 32
        // then spreads them in as many as necessary

        while( numberOfThreads + threadsPerRow >= this->warpSize )
        {
            // counts number of usable threads
            IndexType usedThreads = this->warpSize - numberOfThreads;

            // if this condition applies we can finish the warp
            if( usedThreads == threadsPerRow )
            {
                localLoad = max( localLoad, roundUpDivision( rowLength, threadsPerRow ) );

                reduceMap[ numberOfThreads ] = 1;
                for( IndexType i = numberOfThreads + 1; i < this->warpSize; i++ )
                    reduceMap[ i ] = 0;

                if( !list->addWarp( offset, rowOffset, localLoad, reduceMap ) )
                    return false;

                offset += this->warpSize * localLoad;
                rowOffset = row + 1;

                // IMPORTANT TO RESET VARIABLES
                localLoad = 0;
                for( IndexType i = 0; i < this->warpSize; i++ )
                    reduceMap[ i ] = 0;
                numberOfThreads = 0;
                threadsPerRow = 0;
                addedWarp = true;
		warpId++;
            }

            // if it doesnt apply and if local load isn't equal 0 it will use old local load ( for better balance )
            else if( localLoad != 0 )
            {
                // subtract unmber of used elements and number of used threads
                rowLength -= localLoad * usedThreads;

                threadsPerRow = ( IndexType ) ( rowLength / ave ) + ( rowLength % ave == 0 ? 0 : 1 );

                // fill the reduction map
                reduceMap[ numberOfThreads ] = 1;
                for( IndexType i = numberOfThreads + 1; i < this->warpSize; i++ )
                    reduceMap[ i ] = 0;

                // count new offsets, add new warp and reset variables
                if( !list->addWarp( offset, rowOffset, localLoad, reduceMap ) )
                    return false;
                offset += this->warpSize * localLoad;
                rowOffset = row;

                // RESET VARIABLES
                localLoad = 0;
                for( IndexType i = 0; i < this->warpSize; i++ )
                    reduceMap[ i ] = 0;
                numberOfThreads = 0;
                addedWarp = true;
		warpId++;
            }
            // otherwise we are starting a new warp
            else
            {
                threadsPerRow = ( IndexType ) ( rowLength / ave ) + ( rowLength % ave == 0 ? 0 : 1 );
                if( threadsPerRow < this->warpSize )                
                    break;

                localLoad = ave;
            }
        }
	if( threadsPerRow <= 0 )
	{
	    threadsPerRow = 1;
	    continue;
	}
        localLoad = max( localLoad, roundUpDivision( rowLength, threadsPerRow ) );
        reduceMap[ numberOfThreads ] = 1;
        for( IndexType i = numberOfThreads + 1; i < numberOfThreads + threadsPerRow; i++ )
            reduceMap[ i ] = 0;

        numberOfThreads += threadsPerRow;

        // if last row doesnt fill the whole warp or it fills more warps and threads still remain
        if( ( ( row == this->getRows() - 1 ) && !addedWarp ) ||
            ( ( row == this->getRows() - 1 ) && ( threadsPerRow == numberOfThreads ) && ( numberOfThreads > 0 ) ) )
        {
            list->addWarp( offset, rowOffset, localLoad, reduceMap );
        }
    }
    return true;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlAdEllpackMatrix< Real, Device, Index >::computeWarps( const IndexType SMs,
                                                              const IndexType threadsPerSM,
                                                              warpList* list )
{
    IndexType averageLoad = 0;
    warpInfo* temp = list->getHead()->next;
    while( temp->next != list->getTail() )
    {
        averageLoad += temp->localLoad;
        temp = temp->next;
    }
    averageLoad /= list->getNumberOfWarps();

    IndexType totalWarps = SMs * ( threadsPerSM / this->warpSize );
    IndexType remainingThreads = ( list->getNumberOfWarps() * this->warpSize ) % totalWarps;
    bool warpsToSplit = true;

    while( remainingThreads < ( totalWarps / 2 ) && warpsToSplit )
    {
        warpsToSplit = false;
        temp = list->getHead()->next;
        while( temp != list->getTail() )
        {
            if( temp->localLoad > averageLoad )
            {
                temp = list->splitInHalf( temp );
                warpsToSplit = true;
		
            }
            temp = temp->next;
        }
    }
}

template< typename Real,
          typename Device,
          typename Index >
bool tnlAdEllpackMatrix< Real, Device, Index >::createArrays( warpList* list )
{
    IndexType length = list->getNumberOfWarps();

    if( !this->offset.setSize( length + 1 ) ||
        !this->rowOffset.setSize( length + 1 ) ||
        !this->localLoad.setSize( length ) ||
        !this->reduceMap.setSize( length * this->warpSize ) )
        return false;

    IndexType iteration = 0;
    warpInfo* warp = list->getHead()->next;
    while( warp != list->getTail() )
    {
        this->offset.setElement( iteration, warp->offset );
        this->rowOffset.setElement( iteration, warp->rowOffset );
        this->localLoad.setElement( iteration, warp->localLoad );
        for( int i = iteration * this->warpSize + 1; i < ( iteration + 1 ) * this->warpSize; i++ )
            this->reduceMap.setElement( i, warp->reduceMap[ i & ( this->warpSize - 1 ) ] );
        iteration++;
        warp = warp->next;
    }
    this->rowOffset.setElement( length, this->getRows() );
    this->offset.setElement( length, this->offset.getElement( length - 1 ) + this->warpSize * this->localLoad.getElement( length - 1 ) );
    this->allocateMatrixElements( this->offset.getElement( length ) );

    return true;
}

template<>
class tnlAdEllpackMatrixDeviceDependentCode< tnlHost >
{
public:

    typedef tnlHost Device;

    template< typename Real,
              typename Index,
              typename InVector,
              typename OutVector >
    static void vectorProduct( const tnlAdEllpackMatrix< Real, Device, Index >& matrix,
                               const InVector& inVector,
                               OutVector& outVector )
    {
        for( Index warp = 0; warp < matrix.localLoad.getSize(); warp++ )
	{
	    for( Index i = warp * matrix.warpSize; i < ( warp + 1 ) * matrix.warpSize; i++ )
            {
                Index elementPtr = warp * matrix.warpSize + i;
                Real partialResult = 0.0;
                for( Index j = 0; j < matrix.localLoad.getElement( warp ); j++ )
                {
                    partialResult += matrix.values.getElement( elementPtr ) * inVector[ matrix.columnIndexes.getElement( elementPtr ) ];
                }
                outVector[ matrix.reduceMap.getElement( i ) ] += partialResult;
            }
	}
    }

};

#ifdef HAVE_CUDA
template< typename Real,
          typename Device,
          typename Index >
template< typename InVector,
          typename OutVector >
__device__
void tnlAdEllpackMatrix< Real, Device, Index >::spmvCuda( const InVector& inVector,
                                                          OutVector& outVector,
                                                          const int gridIdx )
{
    IndexType globalIdx = ( gridIdx * tnlCuda::GetMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
    IndexType warpIdx = globalIdx >> 5;
    IndexType inWarpIdx = globalIdx & ( this->warpSize - 1 );
    if( globalIdx > this->reduceMap.getSize() )
	return;

    const blockSize = 256;
    Real* temp = getSharedMemory< Real >();
    __shared__ IndexType reduceMap[ blockSize ];
    reduceMap[ threadIdx.x ] = this->reduceMap[ globalIdx ];
    temp[ threadIdx.x ] = 0.0;
    
    IndexType elementPtr = ( warpIdx << 5 ) + inWarpIdx;
    for( IndexType i = 0; i < this->localLoad[ warpIdx ]; i++ )
    {
	if( this->columnIndexes[ elementPtr ] != this->getPaddingIndex() )
	    temp[ threadIdx.x ] += this->values[ elementPtr ] * inVector[ this->columnIndexes[ elementPtr ] ];
    }
    
    if( reduceMap[ threadIdx.x ] == 1 )
    {
	elementPtr = threadIdx.x + 1;
        while( elementPtr < blockSize && reduceMap[ elementPtr ] == 0 )
		temp[ threadIdx.x ] += temp[ elementPtr ];
        atomicAdd(  );
    }
} 

template< typename Real,
          typename Index,
          typename InVector,
          typename OutVector >
__global__
void tnlAdEllpackMatrixVectorProductCuda( const tnlAdEllpackMatrix< Real, tnlCuda, Index >& matrix,
                                          const InVector& inVector,
                                          OutVector& outVector,
                                          const int gridIdx )
{
    matrix->spmvCuda( inVector, outVector, gridIdx );
}

template<>
class tnlAdEllpackMatrixDeviceDependentCode< tnlCuda >
{
public:

    typedef tnlCuda Device;

    template< typename Real,
              typename Index,
              typename InVector,
              typename OutVector >
    static void vectorProduct( const tnlAdEllpackMatrix< Real, Device, Index >& matrix,
                               const InVector& inVector,
                               OutVector& outVector )
    {
        typedef tnlAdEllpackMatrix< Real, tnlCuda, Index > Matrix;
	typedef Index IndexType;
	Matrix kernel_this = tnlCuda::passToDevice( matrix );
	InVector* kernel_inVector = tnlCuda::passToDevice( inVector );
	OutVector* kernel_outVector = tnlCuda::pasToDevice( outVector );
	dim3 blockSize( 256 ), cudaGridSize( tnlCuda::getMaxGridSize() );
	dim3 cudaBlocks = roundUpDivision( matrix.reduceMap.getSize(), blockSize );
	dim3 cudaGrids = roundUpDivision( cudaBlocks, tnlCuda::getMaxGridSize() );
	for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ );
	{
	    if( gridIdx == cudaGrids - 1 )
		cudaGridSize.x = cudaBlocks % tnlCuda::getMaxGridSize();
	    const int sharedMemory = blockSize.x * sizeof( Real );
	    tnlAdEllpackMatrixVectorProductCuda< Real, Index, InVector, OutVector >
                                               <<< cudaGridSize, blockSize, sharedMemory >>>
                                               ( kernel_this,
                                                 kernel_inVector,
                                                 kernel_outVector,
                                                 gridIdx );
	}
	tnlCuda::freeFromDevice( kernel_this );
	tnlCuda::freeFromDevice( kernel_inVector );
	tnlCuda::freeFromDevice( kernel_outVector );
	checkCudaDevice;
    }

};
#endif


#endif /*_TNLADELLPACKMATRIX_IMPL_H_*/
