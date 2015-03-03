#ifndef TNLBIELLPACKMATRIX_IMPL_H_
#define TNLBIELLPACKMATRIX_IMPL_H_

#include <matrices/tnlBiEllpackMatrix.h>
#include <core/vectors/tnlVector.h>
#include <core/vectors/tnlSharedVector.h>
#include <core/mfuncs.h>

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
#ifdef HAVE_CUDA
__device__ __host__
#endif
Index tnlBiEllpackMatrix< Real, Device, Index, StripSize >::power( const IndexType number,
																   const IndexType exponent ) const
{
	if( exponent >= 0 )
	{
		IndexType result = 1;
		for( IndexType i = 0; i < exponent; i++ )
			result *= number;
		return result;
	}
	return 0;
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
tnlBiEllpackMatrix< Real, Device, Index, StripSize >::tnlBiEllpackMatrix()
: warpSize( 32 ),
  logWarpSize( 5 )
{}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
tnlString tnlBiEllpackMatrix< Real, Device, Index, StripSize >::getType()
{
	return tnlString( "BiEllpackMatrix< ") +
	       tnlString( ::getType< Real >() ) +
	       tnlString( ", " ) +
	       Device :: getDeviceType() +
	       tnlString( " >" );
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
tnlString tnlBiEllpackMatrix< Real, Device, Index, StripSize >::getTypeVirtual() const
{
	return this->getType();
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
bool tnlBiEllpackMatrix< Real, Device, Index, StripSize >::setDimensions( const IndexType rows,
															   	   	   	  const IndexType columns )
{
	tnlAssert( rows >= 0 && columns >= 0,
			   cerr << "rows = " << rows
			   	    << "columns = " <<columns <<endl );

	if( this->getRows() % this->warpSize != 0 )
		this->setVirtualRows( this->getRows() + this->warpSize - ( this->getRows() % this->warpSize ) );
	else
		this->setVirtualRows( this->getRows() );
	IndexType strips = this->virtualRows / this->warpSize;

	if( ! tnlSparseMatrix< Real, Device, Index >::setDimensions( rows, columns ) ||
        ! this->rowPermArray.setSize( this->rows ) ||
	    ! this->groupPointers.setSize( strips * ( this->logWarpSize + 1 ) + 1 ) )
	    return false;

	for( IndexType row = 0; row < this->getRows(); row++ )
		this->rowPermArray.setElement(row, row);
	return true;
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
bool tnlBiEllpackMatrix< Real, Device, Index, StripSize >::setRowLengths(const RowLengthsVector& rowLengths)
{
	if( this->getRows() % this->warpSize != 0 )
		this->setVirtualRows( this->getRows() + this->warpSize - ( this->getRows() % this->warpSize ) );
	else
		this->setVirtualRows( this->getRows() );
	IndexType strips = this->virtualRows / this->warpSize;
	if( ! this->rowPermArray.setSize( this->rows ) ||
		! this->groupPointers.setSize( strips * ( this->logWarpSize + 1 ) + 1 ) )
		return false;
	for( IndexType i = 0; i < this->groupPointers.getSize(); i++ )
		this->groupPointers.setElement( i, 0 );

	DeviceDependentCode::performRowBubbleSort( *this, /*rowLengths,*/ rowLengths );
	DeviceDependentCode::verifyRowPerm( *this, /*rowLengths,*/ rowLengths );

	DeviceDependentCode::computeColumnSizes( *this, /*tempRowLengths*/ rowLengths );
	this->groupPointers.computeExclusivePrefixSum();
	cout << "verifying row lengths" << endl;
	//cout << "setRowLengths " << this->groupPointers.getElement( 0 ) << "    " << this->groupPointers.getElement( 1 ) << endl;
	DeviceDependentCode::verifyRowLengths( *this, rowLengths );

	return
		this->allocateMatrixElements( this->warpSize * this->groupPointers.getElement( strips * ( this->logWarpSize + 1 ) ) );
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
#ifdef HAVE_CUDA
__device__ __host__
#endif
Index tnlBiEllpackMatrix< Real, Device, Index, StripSize >::getStripLength( const IndexType strip ) const
{
	tnlAssert( strip >= 0,
				cerr << "strip = " << strip
					 << " this->getName() = " << this->getName() << endl );

	return this->groupPointers[ ( strip + 1 ) * ( this->logWarpSize + 1 ) ]
	                            - this->groupPointers[ strip * ( this->logWarpSize + 1 ) ];
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
#ifdef HAVE_CUDA
__device__ __host__
#endif
Index tnlBiEllpackMatrix< Real, Device, Index, StripSize >::getNumberOfGroups( const IndexType row ) const
{
	tnlAssert( row >=0 && row < this->getRows(),
	              cerr << "row = " << row
	                   << " this->getRows() = " << this->getRows()
	                   << " this->getName() = " << this->getName() << endl );

	const IndexType strip = row / this->warpSize;
	const IndexType rowStripPermutation = this->rowPermArray[ row ] - this->warpSize * strip;
	const IndexType numberOfGroups = this->logWarpSize + 1;
	for( IndexType i = 0; i < this->logWarpSize + 1; i++ )
		if( rowStripPermutation < this->power( 2, i ) )
			return ( numberOfGroups - i );
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
Index tnlBiEllpackMatrix< Real, Device, Index, StripSize >::getRowLength( const IndexType row ) const
{
	tnlAssert( row >= 0 && row < this->getRows(),
				cerr << "row = " << row
					 << " this->getRows() = " << this->getRows()
					 << " this->getName() = " << this->getName() << endl );

	const IndexType strip = row / this->warpSize;
	const IndexType stripLength = this->getStripLength( strip );
	const IndexType groupBegin = strip * ( this->logWarpSize + 1 );
	const IndexType rowStripPerm = this->rowPermArray.getElement( row ) - strip * this->warpSize;
	const IndexType begin = this->groupPointers.getElement( groupBegin ) * this->warpSize + rowStripPerm * stripLength;
	IndexType elementPtr = begin;

	IndexType rowLength = 0;

	for( IndexType group = 0; group < this->getNumberOfGroups( row ); group++ )
	{
		for( IndexType i = 0; i < this->getGroupLength( strip, group ); i++ )
		{
			IndexType biElementPtr = elementPtr;
			for( IndexType j = 0; j < this->power( 2, group ); j++ )
			{
				if( this->values.getElement( biElementPtr ) == 0.0 )
					return rowLength;
				else
					rowLength++;
				biElementPtr += this->power( 2, this->logWarpSize - group ) * stripLength;
			}
			elementPtr++;
		}
	}
	return rowLength;
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
	template< typename Real2,
			  typename Device2,
			  typename Index2 >
bool tnlBiEllpackMatrix< Real, Device, Index, StripSize >::setLike( const tnlBiEllpackMatrix< Real2, Device2, Index2, StripSize >& matrix )
{
	cout << "setLike" << endl;
	cout << "settingLike" << endl;
	if( ! tnlSparseMatrix< Real, Device, Index >::setLike( matrix ) ||
		! this->rowPermArray.setLike( matrix.rowPermArray ) ||
		! this->groupPointers.setLike( matrix.groupPointers ) )
		return false;
	return true;
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
void tnlBiEllpackMatrix< Real, Device, Index, StripSize >::getRowLengths( tnlVector< IndexType, DeviceType, IndexType >& rowLengths) const
{
	for( IndexType row = 0; row < this->getRows(); row++ )
		rowLengths.setElement( row, this->getRowLength( row ) );
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
bool tnlBiEllpackMatrix< Real, Device, Index, StripSize >::setElement( const IndexType row,
																	   const IndexType column,
																	   const RealType& value )
{
	tnlAssert( ( row >=0 && row < this->getRows() ) ||
			    ( column >= 0 && column < this->getColumns() ),
	              cerr << "row = " << row
	                   << " this->getRows() = " << this->getRows()
	                   << " this->getColumns() = " << this->getColumns()
	                   << " this->getName() = " << this->getName() << endl );

	return this->addElement( row, column, value, 0.0 );
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
bool tnlBiEllpackMatrix< Real, Device, Index, StripSize >::addElement( const IndexType row,
																	   const IndexType column,
																	   const RealType& value,
																	   const RealType& thisElementMultiplicator )
{
	const IndexType strip = row / this->warpSize;
	const IndexType stripLength = this->getStripLength( strip );
	const IndexType groupBegin = strip * ( this->logWarpSize + 1 );
	const IndexType rowStripPerm = this->rowPermArray.getElement( row ) - strip * this->warpSize;
	const IndexType begin = this->groupPointers.getElement( groupBegin ) * this->warpSize + rowStripPerm * stripLength;
	IndexType elementPtr = begin;

	for( IndexType group = 0; group < this->getNumberOfGroups( row ); group++ )
	{
		for( IndexType i = 0; i < this->getGroupLength( strip, group ); i++ )
		{
			IndexType biElementPtr = elementPtr;
			for( IndexType j = 0; j < this->power( 2, group ); j++ )
			{
				if( this->columnIndexes.getElement( biElementPtr ) == this->getPaddingIndex() )
				{
					this->columnIndexes.setElement( biElementPtr, column );
					this->values.setElement( biElementPtr, value );
					return true;
				}
				if( this->columnIndexes.getElement( biElementPtr ) == column )
				{
					this->values.setElement( biElementPtr, this->values.getElement( biElementPtr ) + value * thisElementMultiplicator );
					return true;
				}
				biElementPtr += this->power( 2, this->logWarpSize - group ) * stripLength;
			}
			elementPtr++;
		}
	}
	cout << "nepovedlo se" << endl;
	return false;
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
bool tnlBiEllpackMatrix< Real, Device, Index, StripSize >::setRow( const IndexType row,
																   const IndexType* columns,
																   const RealType* values,
																   const IndexType numberOfElements )
{
	tnlAssert( row >= 0 && row < this->getRows(),
						cerr << "row = " << row
							 << " this->getRows() = " << this->getRows()
							 << " this->getName() = " << this->getName() << endl );

	const IndexType strip = row / this->warpSize;
	const IndexType stripLength = this->getStripLength( strip );
	const IndexType groupBegin = strip * ( this->logWarpSize + 1 );
	const IndexType rowStripPerm = this->rowPermArray.getElement( row ) - strip * this->warpSize;
	const IndexType begin = this->groupPointers.getElement( groupBegin ) * this->warpSize + rowStripPerm * stripLength;
	IndexType elementPtr = begin;

	IndexType thisElementPtr = 0;

	for( IndexType group = 0; ( group < this->getNumberOfGroups( row ) ) && ( thisElementPtr < numberOfElements ); group++ )
	{
		for( IndexType i = 0; i < this->getGroupLength( strip, group ); i++ )
		{
			IndexType biElementPtr = elementPtr;
			for( IndexType j = 0; ( j < this->power( 2, group ) ) && ( thisElementPtr < numberOfElements ) ; j++ )
			{
				this->columnIndexes.setElement( biElementPtr, columns[ thisElementPtr ] );
				this->values.setElement( biElementPtr, values[ thisElementPtr ] );
				thisElementPtr++;
				biElementPtr += this->power( 2, this->logWarpSize - group ) * stripLength;
			}
			elementPtr++;
		}
	}
	if( thisElementPtr == numberOfElements )
		return true;
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
bool tnlBiEllpackMatrix< Real, Device, Index, StripSize >::addRow( const IndexType row,
																   const IndexType* columns,
																   const RealType* values,
																   const IndexType numberOfElements,
																   const RealType& thisElementMultiplicator )
{
	tnlAssert( row >=0 && row < this->getRows(),
		              cerr << "row = " << row
		                   << " this->getRows() = " << this->getRows()
		                   << " this->getName() = " << this->getName() << endl );

	bool adding = true;
	const IndexType strip = row / this->warpSize;
	const IndexType stripLength = this->getStripLength( strip );
	const IndexType groupBegin = strip * ( this->logWarpSize + 1 );
	const IndexType rowStripPerm = this->rowPermArray.getElement( row ) - this->warpSize * strip;
	const IndexType begin = this->groupPointers.getElement( groupBegin ) * this->warpSize + rowStripPerm * stripLength;
	IndexType biElementPtr, elementPtr = begin;
	IndexType thisElementPtr = 0;

	while( adding && thisElementPtr < numberOfElements )
	{
		adding = false;
		for( IndexType group = 0; group < this->getNumberOfGroups( row ); group++ )
		{
			for( IndexType i = 0; i < this->getGroupLength( strip, group ); i++ )
			{
				biElementPtr = elementPtr;
				for( IndexType j = 0; ( j < this->power( 2, group ) ) && ( thisElementPtr < numberOfElements ); j++ )
				{
					if( this->columnIndexes.getElement( biElementPtr ) == columns[ thisElementPtr ] )
					{
						RealType result = this->values.getElement( biElementPtr ) + values[ thisElementPtr ] * thisElementMultiplicator;
						this->values.setElement( biElementPtr, result );
						thisElementPtr++;
					}
					biElementPtr += this->power( 2, this->logWarpSize - group ) * stripLength;
				}
				elementPtr++;
			}
		}
	}
	return ( thisElementPtr == numberOfElements );
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
Real tnlBiEllpackMatrix< Real, Device, Index, StripSize >::getElement( const IndexType row,
																	   const IndexType column ) const
{
	tnlAssert( ( row >= 0 && row < this->getRows() ) ||
				( column >= 0 && column < this->getColumns() ),
				  cerr << "row = " << row
				  	   << " this->getRows() = " << this->getRows()
				  	   << " this->getColumns() = " << this->getColumns()
				  	   << "this->getName() = " << this->getName() << endl );

	const IndexType strip = row / this->warpSize;
	const IndexType stripLength = this->getStripLength( strip );
	const IndexType groupBegin = strip * ( this->logWarpSize + 1 );
	const IndexType rowStripPerm = this->rowPermArray.getElement( row ) - strip * this->warpSize;
	const IndexType begin = this->groupPointers.getElement( groupBegin ) * this->warpSize + rowStripPerm * stripLength;
	IndexType elementPtr = begin;

	for( IndexType group = 0; group < this->getNumberOfGroups( row ); group++ )
	{
		for( IndexType i = 0; i < this->getGroupLength( strip, group ); i++ )
		{
			IndexType biElementPtr = elementPtr;
			for( IndexType j = 0; j < this->power( 2, group ); j++ )
			{
				if( this->columnIndexes.getElement( biElementPtr ) == column )
					return this->values.getElement( biElementPtr );
				biElementPtr += this->power( 2, this->logWarpSize - group ) * stripLength;
			}
			elementPtr++;
		}
	}
	return 0.0;
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
void tnlBiEllpackMatrix< Real, Device, Index, StripSize >::getRow( const IndexType row,
																   IndexType* columns,
																   RealType* values ) const
{
	tnlAssert( row >=0 && row < this->getRows(),
	              cerr << "row = " << row
	                   << " this->getRows() = " << this->getRows()
	                   << " this->getName() = " << this->getName() << endl );

	bool padding = false;
	const IndexType strip = row / this->warpSize;
	const IndexType stripLength = this->getStripLength( strip );
	const IndexType groupBegin = strip * ( this->logWarpSize + 1 );
	const IndexType rowStripPerm = this->rowPermArray.getElement( row ) - this->warpSize * strip;
	const IndexType begin = this->groupPointers.getElement( groupBegin ) * this->warpSize + rowStripPerm * stripLength;
	IndexType elementPtr = begin;

	IndexType thisElementPtr = 0;
	for( IndexType group = 0; group < this->getNumberOfGroups( row ) && !padding; group++ )
	{
		for( IndexType i = 0; i < this->getGroupLength( strip, group ); i++ )
		{
			IndexType biElementPtr = elementPtr;
			for( IndexType j = 0; j < this->power( 2, group ) && !padding; j++ )
			{
				if( this->columnIndexes.getElement( biElementPtr ) == this->getPaddingIndex() )
					padding = true;
				else
				{
					values[ thisElementPtr ] = this->values.getElement( biElementPtr );
					columns[ thisElementPtr ] = this->columnIndexes.getElement( biElementPtr );
					thisElementPtr++;
				}
				biElementPtr += this->power( 2, this->logWarpSize - group ) * stripLength;
			}
			elementPtr++;
		}
	}
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
void tnlBiEllpackMatrix< Real, Device, Index, StripSize >::setVirtualRows(const IndexType rows)
{
	this->virtualRows = rows;
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
#ifdef HAVE_CUDA
__device__ __host__
#endif
Index tnlBiEllpackMatrix< Real, Device, Index, StripSize >::getGroupLength( const Index strip,
																 	 	    const Index group ) const
{
	return this->groupPointers[ strip * ( this->logWarpSize + 1 ) + group + 1 ]
			- this->groupPointers[ strip * ( this->logWarpSize + 1 ) + group ];
}

template< typename Real,
	  	  typename Device,
	  	  typename Index,
	  	  int StripSize >
template< typename InVector,
	  	  typename OutVector >
void tnlBiEllpackMatrix< Real, Device, Index, StripSize >::vectorProduct( const InVector& inVector,
										  	  	  	  		   	   	      OutVector& outVector ) const
{
	DeviceDependentCode::vectorProduct( *this, inVector, outVector );
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
template< typename InVector,
		  typename OutVector >
void tnlBiEllpackMatrix< Real, Device, Index, StripSize >::vectorProductHost( const InVector& inVector,
																			  OutVector& outVector ) const
{
	const IndexType numberOfStrips = this->virtualRows / this->warpSize;
	for( IndexType strip = 0; strip < numberOfStrips; strip++ )
	{
		const IndexType stripLength = this->getStripLength( strip );
		const IndexType groupBegin = strip * ( this->logWarpSize + 1 );
		const IndexType stripBegin = strip * this->warpSize;
		const IndexType stripEnd = stripBegin + this->warpSize;
		IndexType end = stripEnd;
		if( end > this->getRows() )
			end = this->getRows();

		tnlVector< Real, Device, Index > tempStripOutVector;
		tempStripOutVector.setSize( end - stripBegin );
		for( IndexType i = 0; i < tempStripOutVector.getSize(); i++ )
			tempStripOutVector.setElement( i, 0 );

		for( IndexType row = stripBegin; row < stripEnd; row++ )
		{
			IndexType currentRow = row;
			IndexType elementPtr = this->groupPointers.getElement( groupBegin ) * this->warpSize + ( row - stripBegin ) * stripLength;
			for( IndexType group = 0; group < this->logWarpSize + 1; group++ )
			{
				if( !( currentRow - stripBegin < this->power( 2, logWarpSize - group ) ) )
					currentRow -= this->power( 2, this->logWarpSize - group );
				for( IndexType i = 0; i < this->getGroupLength( strip, group ); i++ )
				{
					if( this->columnIndexes.getElement( elementPtr ) == this->getPaddingIndex() )
					{
						elementPtr++;
						continue;
					}
					RealType result = tempStripOutVector.getElement( currentRow - stripBegin );
					result += inVector[ this->columnIndexes.getElement( elementPtr ) ] * this->values.getElement( elementPtr );
					tempStripOutVector.setElement( currentRow - stripBegin, result );
					elementPtr++;
				}
			}
		}

		for( IndexType i = stripBegin; i < end; i++ )
			outVector[ i ] = tempStripOutVector.getElement( this->rowPermArray.getElement( i ) - stripBegin );
		tempStripOutVector.reset();
	}
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
void tnlBiEllpackMatrix< Real, Device, Index, StripSize >::reset()
{
	tnlSparseMatrix< Real, Device, Index >::reset();
	this->rowPermArray.reset();
	this->groupPointers.reset();
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
bool tnlBiEllpackMatrix< Real, Device, Index, StripSize >::save( tnlFile& file ) const
{
	if( ! tnlSparseMatrix< Real, Device, Index >::save( file ) ||
		! this->groupPointers.save( file ) ||
		! this->rowPermArray.save( file ) )
		return false;
	return true;
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
bool tnlBiEllpackMatrix< Real, Device, Index, StripSize >::load( tnlFile& file )
{
	if( ! tnlSparseMatrix< Real, Device, Index >::load( file ) ||
		! this->groupPointers.load( file ) ||
		! this->rowPermArray.load( file ) )
		return false;
	return true;
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
bool tnlBiEllpackMatrix< Real, Device, Index, StripSize >::save( const tnlString& fileName ) const
{
	return tnlObject::save( fileName );
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
bool tnlBiEllpackMatrix< Real, Device, Index, StripSize >::load( const tnlString& fileName )
{
	return tnlObject::load( fileName );
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
void tnlBiEllpackMatrix< Real, Device, Index, StripSize >::print( ostream& str ) const
{
	for( IndexType row = 0; row < this->getRows(); row++ )
	{
		str <<"Row: " << row << " -> ";
		bool padding = false;
		const IndexType strip = row / this->warpSize;
		const IndexType stripLength = this->getStripLength( strip );
		const IndexType groupBegin = strip * ( this->logWarpSize + 1 );
		const IndexType rowStripPerm = this->rowPermArray.getElement( row ) - this->warpSize * strip;
		const IndexType begin = this->groupPointers.getElement( groupBegin ) * this->warpSize + rowStripPerm * stripLength;
		IndexType elementPtr = begin;

		for( IndexType group = 0; group < this->getNumberOfGroups( row ) && !padding; group++ )
		{
			for( IndexType i = 0; i < this->getGroupLength( strip, group ) && !padding; i++ )
			{
				IndexType biElementPtr = elementPtr;
				for( IndexType j = 0; j < this->power( 2, group ) && !padding; j++ )
				{
					if( this->columnIndexes.getElement( biElementPtr ) == this->getPaddingIndex() )
						padding = true;
					RealType value = this->values.getElement( biElementPtr );
					IndexType column = this->columnIndexes.getElement( biElementPtr );
					str << " Col:" << column << "->" << value << "\t";
					biElementPtr += this->power( 2, this->logWarpSize - group ) * stripLength;
				}
			}
		}
		str << endl;
	}
}

/*template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
void tnlBiEllpackMatrix< Real, Device, Index, StripSize >::verifyRowLengths( const typename tnlBiEllpackMatrix< Real, Device, Index, StripSize >::RowLengthsVector& rowLengths )
{
	bool ok = true;
	cout << "inside method" << endl;
	for( IndexType row = 0; row < this->getRows(); row++ )
	{
		const IndexType strip = row / this->warpSize;
		const IndexType stripLength = this->getStripLength( strip );
		const IndexType groupBegin = ( this->logWarpSize + 1 ) * strip;
		const IndexType rowStripPerm = this->rowPermArray.getElement( row ) - strip * this->warpSize;
		const IndexType begin = this->groupPointers.getElement( groupBegin ) * this->warpSize + rowStripPerm * stripLength;
		IndexType elementPtr = begin;
		IndexType rowLength = 0;

		for( IndexType group = 0; group < this->getNumberOfGroups( row ); group++ )
		{
			for( IndexType i = 0; i < this->getGroupLength( strip, group ); i++ )
			{
				IndexType biElementPtr = elementPtr;
				for( IndexType j = 0; j < this->power( 2, group ); j++ )
				{

					rowLength++;
					biElementPtr += this->power( 2, this->logWarpSize - group ) * stripLength;
				}
				elementPtr++;
			}
		}
		if( rowLengths.getElement( row ) > rowLength )
			ok = false;
	}
	if( ok )
		cout << "row lengths OK" << endl;
}*/

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
void tnlBiEllpackMatrix< Real, Device, Index, StripSize >::performRowBubbleSort( tnlVector< Index, Device, Index >& tempRowLengths )
{
	//TODO: da se zrychlit?
	Index strips = this->virtualRows / this->warpSize;
	for( Index i = 0; i < strips; i++ )
	{
		Index begin = i * this->warpSize;
		Index end = ( i + 1 ) * this->warpSize - 1;
		if( this->getRows() - 1 < end)
			end = this->getRows() - 1;
		bool sorted = false;
		Index permIndex1, permIndex2, offset = 0;
		while( !sorted )
		{
			sorted = true;
			for( Index j = begin + offset; j < end - offset; j++ )
				if( tempRowLengths.getElement( j ) < tempRowLengths.getElement( j + 1 ) )
				{
					for( Index k = begin; k < end + 1; k++ )
					{
						if( this->rowPermArray.getElement( k ) == j )
							permIndex1 = k;
						if( this->rowPermArray.getElement( k ) == j + 1 )
							permIndex2 = k;
					}
					Index temp = tempRowLengths.getElement( j );
					tempRowLengths.setElement( j, tempRowLengths.getElement( j + 1 ) );
					tempRowLengths.setElement( j + 1, temp );
					temp = this->rowPermArray.getElement( permIndex1 );
					this->rowPermArray.setElement( permIndex1, this->rowPermArray.getElement( permIndex2 ) );
					this->rowPermArray.setElement( permIndex2, temp );
					sorted = false;
				}
			for( Index j = end - 1 - offset; j > begin + offset; j-- )
				if( tempRowLengths.getElement( j ) > tempRowLengths.getElement( j - 1 ) )
				{
					for( Index k = begin; k < end + 1; k++ )
					{
						if( this->rowPermArray.getElement( k ) == j )
							permIndex1 = k;
						if( this->rowPermArray.getElement( k ) == j - 1 )
							permIndex2 = k;
					}
					Index temp = tempRowLengths.getElement( j );
					tempRowLengths.setElement( j, tempRowLengths.getElement( j - 1 ) );
					tempRowLengths.setElement( j - 1, temp );
					temp = this->rowPermArray.getElement( permIndex1 );
					this->rowPermArray.setElement( permIndex1, this->rowPermArray.getElement( permIndex2 ) );
					this->rowPermArray.setElement( permIndex2, temp );
					sorted = false;
				}
			offset++;
		}
	}
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
void tnlBiEllpackMatrix< Real, Device, Index, StripSize >::computeColumnSizes( tnlVector< Index, Device, Index >& tempRowLengths )
{
	Index numberOfStrips = this->virtualRows / this->warpSize;
	for( Index strip = 0; strip < numberOfStrips; strip++ )
	{
		Index i = 0;
		Index rowBegin = strip * this->warpSize;
		Index groupBegin = strip * ( this->logWarpSize + 1 );
		Index emptyGroups = 0;
		if( strip == numberOfStrips - 1 )
		{
			Index lastRows = this->getRows() - rowBegin;
			while( !( lastRows > this->power( 2, this->logWarpSize - 1 - emptyGroups ) ) )
				emptyGroups++;
			for( Index group = groupBegin; group < groupBegin + emptyGroups; group++ )
				this->groupPointers.setElement( group, 0 );
		}
		i += emptyGroups;
		for( Index group = groupBegin + emptyGroups; group < groupBegin + this->logWarpSize; group++ )
		{
			Index row = this->power( 2, 4 - i );
			Index temp = tempRowLengths.getElement( row + rowBegin );
			for( Index prevGroups = groupBegin; prevGroups < group; prevGroups++ )
				temp -= this->power( 2, prevGroups - groupBegin ) * this->groupPointers.getElement( prevGroups );
			temp =  ceil( ( float ) temp / this->power( 2, i ) );
			this->groupPointers.setElement( group, temp );
			i++;
		}
		Index temp = tempRowLengths.getElement( rowBegin );
		for( Index prevGroups = groupBegin; prevGroups < groupBegin + this->logWarpSize; prevGroups++ )
			temp -= this->power( 2, prevGroups - groupBegin ) * this->groupPointers.getElement( prevGroups );
		temp = ceil( ( float ) temp / this->power( 2, this->logWarpSize ) );
		this->groupPointers.setElement( groupBegin + this->logWarpSize, temp );
	}
}

template<>
class tnlBiEllpackMatrixDeviceDependentCode< tnlHost >
{
public:

	typedef tnlHost Device;

	template< typename Real,
				  typename Index,
				  int StripSize >
		static void verifyRowLengths( const tnlBiEllpackMatrix< Real, Device, Index, StripSize >& matrix,
									  const typename tnlBiEllpackMatrix< Real, Device, Index, StripSize >::RowLengthsVector& rowLengths )
		{
			bool ok = true;
			cout << "inside method" << endl;
			for( Index row = 0; row < matrix.getRows(); row++ )
			{
				const Index strip = row / matrix.warpSize;
				const Index stripLength = matrix.getStripLength( strip );
				const Index groupBegin = ( matrix.logWarpSize + 1 ) * strip;
				const Index rowStripPerm = matrix.rowPermArray.getElement( row ) - strip * matrix.warpSize;
				const Index begin = matrix.groupPointers.getElement( groupBegin ) * matrix.warpSize + rowStripPerm * stripLength;
				Index elementPtr = begin;
				Index rowLength = 0;

				for( Index group = 0; group < matrix.getNumberOfGroups( row ); group++ )
				{
					for( Index i = 0; i < matrix.getGroupLength( strip, group ); i++ )
					{
						Index biElementPtr = elementPtr;
						for( Index j = 0; j < matrix.power( 2, group ); j++ )
						{

							rowLength++;
							biElementPtr += matrix.power( 2, matrix.logWarpSize - group ) * stripLength;
						}
						elementPtr++;
					}
				}
				if( rowLengths.getElement( row ) > rowLength )
					ok = false;
			}
			if( ok )
				cout << "row lengths OK" << endl;
		}

	template< typename Real,
			  typename Index,
			  int StripSize >
	static void verifyRowPerm( const tnlBiEllpackMatrix< Real, Device, Index, StripSize >& matrix,
							   const typename tnlBiEllpackMatrix< Real, Device, Index, StripSize >::RowLengthsVector& rowLengths
							   /*tnlVector< Index, Device, Index >& tempRowLengths*/ )
	{
		bool ok = true;
		Index numberOfStrips = matrix.virtualRows / matrix.warpSize;
		for( Index strip = 0; strip < numberOfStrips; strip++ )
		{
			Index begin = strip * matrix.warpSize;
			Index end = ( strip + 1 ) * matrix.warpSize;
			if( matrix.getRows() < end )
				end = matrix.getRows();
			for( Index i = begin; i < end - 1; i++ )
			{
				Index permIndex1, permIndex2;
				bool first = false;
				bool second = false;
				for( Index j = begin; j < end; j++ )
				{
					if( matrix.rowPermArray.getElement( j ) == i )
					{
						permIndex1 = j;
						first = true;
					}
					if( matrix.rowPermArray.getElement( j ) == i + 1 )
					{
						permIndex2 = j;
						second = true;
					}
				}
				if( !first || !second )
					cout << "nenasel jsem spravne indexy" << endl;
				if( rowLengths.getElement( permIndex1 ) >= rowLengths.getElement( permIndex2 ) )
					continue;
				else
					ok = false;
			}
		}
		if( ok )
			cout << "perm OK" << endl;
	}

	template< typename Real,
			  typename Index,
			  int StripSize,
			  typename InVector,
			  typename OutVector >
	static void vectorProduct( const tnlBiEllpackMatrix< Real, Device, Index, StripSize >& matrix,
							   const InVector& inVector,
						       OutVector& outVector )
	{
		matrix.vectorProductHost( inVector, outVector );
	}

	template< typename Real,
			  typename Index,
			  int StripSize >
	static void computeColumnSizes( tnlBiEllpackMatrix< Real, Device, Index, StripSize >& matrix,
			 	 	 	 	 	 	const typename tnlBiEllpackMatrix< Real, Device, Index, StripSize >::RowLengthsVector& rowLengths )
	{
		Index numberOfStrips = matrix.virtualRows / matrix.warpSize;
		for( Index strip = 0; strip < numberOfStrips; strip++ )
		{
			Index i = 0;
			Index rowBegin = strip * matrix.warpSize;
			Index groupBegin = strip * ( matrix.logWarpSize + 1 );
			Index emptyGroups = 0;
			if( strip == numberOfStrips - 1 )
			{
				Index lastRows = matrix.getRows() - rowBegin;
				while( !( lastRows > matrix.power( 2, matrix.logWarpSize - 1 - emptyGroups ) ) )
					emptyGroups++;
				for( Index group = groupBegin; group < groupBegin + emptyGroups; group++ )
					matrix.groupPointers.setElement( group, 0 );
			}
			i += emptyGroups;
			for( Index group = groupBegin + emptyGroups; group < groupBegin + matrix.logWarpSize; group++ )
			{
				Index row = matrix.power( 2, 4 - i );
				Index permRow = 0;
				while( matrix.rowPermArray.getElement( permRow + rowBegin ) != row + rowBegin )
					permRow++;
				Index temp = rowLengths.getElement( permRow + rowBegin );
				for( Index prevGroups = groupBegin; prevGroups < group; prevGroups++ )
					temp -= matrix.power( 2, prevGroups - groupBegin ) * matrix.groupPointers.getElement( prevGroups );
				temp =  ceil( ( float ) temp / matrix.power( 2, i ) );
				matrix.groupPointers.setElement( group, temp );
				i++;
			}
			Index permRow = rowBegin;
			while( matrix.rowPermArray.getElement( permRow ) != rowBegin )
				permRow++;
			Index temp = rowLengths.getElement( permRow );
			for( Index prevGroups = groupBegin; prevGroups < groupBegin + matrix.logWarpSize; prevGroups++ )
				temp -= matrix.power( 2, prevGroups - groupBegin ) * matrix.groupPointers.getElement( prevGroups );
			temp = ceil( ( float ) temp / matrix.power( 2, matrix.logWarpSize ) );
			matrix.groupPointers.setElement( groupBegin + matrix.logWarpSize, temp );
		}
	}

	template< typename Real,
			  typename Index,
			  int StripSize >
	static void performRowBubbleSort( tnlBiEllpackMatrix< Real, Device, Index, StripSize >& matrix,
									  const typename tnlBiEllpackMatrix< Real, Device, Index, StripSize >::RowLengthsVector& rowLengths
							   	   	  /*tnlVector< Index, Device, Index >& tempRowLengths*/ )
	{
		Index strips = matrix.virtualRows / matrix.warpSize;
		for( Index i = 0; i < strips; i++ )
		{
			Index begin = i * matrix.warpSize;
			Index end = ( i + 1 ) * matrix.warpSize - 1;
			if(matrix.getRows() - 1 < end)
				end = matrix.getRows() - 1;
			bool sorted = false;
			Index permIndex1, permIndex2, offset = 0;
			while( !sorted )
			{
				sorted = true;
				for( Index j = begin + offset; j < end - offset; j++ )
				{
					for( Index k = begin; k < end + 1; k++ )
					{
						if( matrix.rowPermArray.getElement( k ) == j )
							permIndex1 = k;
						if( matrix.rowPermArray.getElement( k ) == j + 1 )
							permIndex2 = k;
					}
					if( rowLengths.getElement( permIndex1 ) < rowLengths.getElement( permIndex2 ) )
					{
						Index temp = matrix.rowPermArray.getElement( permIndex1 );
						matrix.rowPermArray.setElement( permIndex1, matrix.rowPermArray.getElement( permIndex2 ) );
						matrix.rowPermArray.setElement( permIndex2, temp );
						sorted = false;
					}
				}
				for( Index j = end - 1 - offset; j > begin + offset; j-- )
				{
					for( Index k = begin; k < end + 1; k++ )
					{
						if( matrix.rowPermArray.getElement( k ) == j )
							permIndex1 = k;
						if( matrix.rowPermArray.getElement( k ) == j - 1 )
							permIndex2 = k;
					}
					if( rowLengths.getElement( permIndex2 ) < rowLengths.getElement( permIndex1 ) )
					{
						Index temp = matrix.rowPermArray.getElement( permIndex1 );
						matrix.rowPermArray.setElement( permIndex1, matrix.rowPermArray.getElement( permIndex2 ) );
						matrix.rowPermArray.setElement( permIndex2, temp );
						sorted = false;
					}
				}
				offset++;
			}
		}
	}
};

#ifdef HAVE_CUDA
template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
template< typename InVector,
		  typename OutVector >
__device__
void tnlBiEllpackMatrix< Real, Device, Index, StripSize >::spmvCuda( const InVector& inVector,
														  	  	  	 OutVector& outVector,
														  	  	  	 const IndexType warpStart,
														  	  	  	 const IndexType warpEnd,
														  	  	  	 const IndexType inWarpIdx ) const
{
	IndexType strip = warpStart / StripSize;
	IndexType groupBegin = strip * ( this->logWarpSize + 1 );
	IndexType end = warpEnd;
	if( end > this->getRows() )
		end = this->getRows();
	Real* temp = getSharedMemory< Real >();
	temp[ threadIdx.x ] = 0.0;
	IndexType row = warpStart + inWarpIdx;
	IndexType currentRow = row;
	for( IndexType group = 0; group < this->logWarpSize + 1; group++ )
	{
		if( !(currentRow - warpStart < this->power( 2, this->logWarpSize - group ) ) )
			currentRow -= this->power( 2, this->logWarpSize - group );
		if( currentRow >= this->getRows() )
			continue;
		IndexType begin = this->groupPointers[ groupBegin + group ] * this->warpSize;
		for( IndexType j = 0; j < this->getGroupLength( strip, group ); j++ )
		{
			IndexType elementPtr = row - warpStart + j * this->warpSize;
			if( this->columnIndexes[ elementPtr ] == this->getPaddingIndex() )
				break;
			RealType result = inVector[ this->columnIndexes[ elementPtr ] ] * this->values[ elementPtr ];
			temp[ currentRow - warpStart ] += result;
		}
	}
	__syncthreads();
	if( row >= this->getRows() )
		return;
	outVector[ row ] = temp[ this->rowPermArray[ row ] - warpStart ];
}
#endif

#ifdef HAVE_CUDA
template< typename Real,
		  typename Index,
		  int StripSize,
		  typename InVector,
		  typename OutVector >
__global__
void tnlBiEllpackMatrixVectorProductCuda( const tnlBiEllpackMatrix< Real, tnlCuda, Index, StripSize >* matrix,
										  const InVector* inVector,
										  OutVector* outVector,
										  int gridIdx,
										  const int warpSize )
{
	Index globalIdx = ( gridIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
	const Index warpStart = warpSize * ( globalIdx / warpSize );
	const Index warpEnd = warpStart + warpSize;
	const Index inWarpIdx = globalIdx % warpSize;
	matrix->spmvCuda( *inVector, *outVector, warpStart, warpEnd, inWarpIdx );
}
#endif

#ifdef HAVE_CUDA
template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
__device__
void tnlBiEllpackMatrix< Real, Device, Index, StripSize >::performRowBubbleSortCudaKernel( const typename tnlBiEllpackMatrix< Real, Device, Index, StripSize >::RowLengthsVector& rowLengths,
																						   const IndexType strip )
{
	IndexType begin = strip * this->warpSize;
	IndexType end = ( strip + 1 ) * this->warpSize - 1;
	if( this->getRows() - 1 < end )
		end = this->getRows() - 1;
	bool sorted = false;
	IndexType permIndex1, permIndex2, offset = 0;
	while( !sorted )
	{
		sorted = true;
		for( IndexType j = begin + offset; j < end - offset; j++ )
		{
			for( IndexType k = begin; k < end + 1; k++)
			{
				if( this->rowPermArray[ k ] == j )
					permIndex1 = k;
				if( this->rowPermArray[ k ] == j + 1 )
					permIndex2 = k;
			}
			if( rowLengths[ permIndex1 ] < rowLengths[ permIndex2 ] )
			{
				IndexType temp = this->rowPermArray[ permIndex1 ];
				this->rowPermArray[ permIndex1 ] = this->rowPermArray[ permIndex2 ];
				this->rowPermArray[ permIndex2 ] = temp;
				sorted = false;
			}
		}
		for( IndexType j = end - 1 - offset; j > begin + offset; j-- )
		{
			for( IndexType k = begin; k < end + 1; k++ )
			{
				if( this->rowPermArray[ k ] == j )
					permIndex1 = k;
				if( this->rowPermArray[ k ] == j - 1)
					permIndex2 = k;
			}
			if( rowLengths[ permIndex2 ] < rowLengths[ permIndex1 ] )
			{
				IndexType temp = this->rowPermArray[ permIndex1 ];
				this->rowPermArray[ permIndex1 ] = this->rowPermArray[ permIndex2 ];
				this->rowPermArray[ permIndex2 ] = temp;
				sorted = false;
			}
		}
		offset++;
	}
}
#endif

#ifdef HAVE_CUDA
template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
__device__
void tnlBiEllpackMatrix< Real, Device, Index, StripSize >::computeColumnSizesCudaKernel( const typename tnlBiEllpackMatrix< Real, Device, Index, StripSize >::RowLengthsVector& rowLengths,
																						 const IndexType numberOfStrips,
																						 const IndexType strip )
{
	if( strip >= numberOfStrips )
		return;
	IndexType i = 0;
	IndexType rowBegin = strip * this->warpSize;
	IndexType groupBegin = strip * ( this->logWarpSize + 1 );
	IndexType emptyGroups = 0;
	if( strip == numberOfStrips - 1 )
	{
		IndexType lastRows = this->getRows() - rowBegin;
		while( !( lastRows > this->power( 2, this->logWarpSize - 1 - emptyGroups ) ) )
			emptyGroups++;
		for( IndexType group = groupBegin; group < groupBegin + emptyGroups; group++ )
			this->groupPointers[ group ] = 0;
	}
	i += emptyGroups;
	for( IndexType group = groupBegin + emptyGroups; group < groupBegin + this->logWarpSize; group++ )
	{
		IndexType row = this->power( 2, 4 - i );
		IndexType permRow = 0;
		while( this->rowPermArray[ permRow + rowBegin ] != row + rowBegin && permRow < this->warpSize )
			permRow++;
		IndexType temp = rowLengths[ permRow + rowBegin ];
		for( IndexType prevGroups = groupBegin; prevGroups < group; prevGroups++ )
			temp -= this->power( 2, prevGroups - groupBegin ) * this->groupPointers[ prevGroups ];
		temp =  ceil( ( float ) temp / this->power( 2, i ) );
		this->groupPointers[ group ] = temp;
		i++;
	}
	IndexType permRow = rowBegin;
	while( this->rowPermArray[ permRow ] != rowBegin && permRow < this->warpSize + rowBegin )
		permRow++;
	IndexType temp = rowLengths[ permRow ];
	for( IndexType prevGroups = groupBegin; prevGroups < groupBegin + this->logWarpSize; prevGroups++ )
		temp -= this->power( 2, prevGroups - groupBegin ) * this->groupPointers[ prevGroups ];
	temp = ceil( ( float ) temp / this->power( 2, this->logWarpSize ) );
	this->groupPointers[ groupBegin + this->logWarpSize ] = temp;
}
#endif

#ifdef HAVE_CUDA
template< typename Real,
		  typename Index,
		  int StripSize >
__global__
void performRowBubbleSortCuda( tnlBiEllpackMatrix< Real, tnlCuda, Index, StripSize >* matrix,
							   const typename tnlBiEllpackMatrix< Real, tnlCuda, Index, StripSize >::RowLengthsVector* rowLengths,
							   int gridIdx )
{
	const Index stripIdx = gridIdx * tnlCuda::getMaxGridSize() * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
	matrix->performRowBubbleSortCudaKernel( *rowLengths, stripIdx );
}
#endif

#ifdef HAVE_CUDA
template< typename Real,
		  typename Index,
		  int StripSize >
__global__
void computeColumnSizesCuda( tnlBiEllpackMatrix< Real, tnlCuda, Index, StripSize >* matrix,
							 const typename tnlBiEllpackMatrix< Real, tnlCuda, Index, StripSize >::RowLengthsVector* rowLengths,
							 const Index numberOfStrips,
							 int gridIdx )
{
	const Index stripIdx = gridIdx * tnlCuda::getMaxGridSize() * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
	matrix->computeColumnSizesCudaKernel( *rowLengths, numberOfStrips, stripIdx );
}
#endif

template<>
class tnlBiEllpackMatrixDeviceDependentCode< tnlCuda >
{
public:

	typedef tnlCuda Device;

	template< typename Real,
			  typename Index,
			  int StripSize >
	static void verifyRowLengths( const tnlBiEllpackMatrix< Real, Device, Index, StripSize >& matrix,
								  const typename tnlBiEllpackMatrix< Real, Device, Index, StripSize >::RowLengthsVector& rowLengths )
	{
		bool ok = true;
		cout << "inside method" << endl;
		for( Index row = 0; row < matrix.getRows(); row++ )
		{
			const Index strip = row / matrix.warpSize;
			const Index stripLength = matrix.getStripLength( strip );
			const Index groupBegin = ( matrix.logWarpSize + 1 ) * strip;
			const Index rowStripPerm = matrix.rowPermArray.getElement( row ) - strip * matrix.warpSize;
			const Index begin = matrix.groupPointers.getElement( groupBegin ) * matrix.warpSize + rowStripPerm * stripLength;
			Index elementPtr = begin;
			Index rowLength = 0;

			for( Index group = 0; group < matrix.getNumberOfGroups( row ); group++ )
			{
				for( Index i = 0; i < matrix.getGroupLength( strip, group ); i++ )
				{
					Index biElementPtr = elementPtr;
					for( Index j = 0; j < matrix.power( 2, group ); j++ )
					{

						rowLength++;
						biElementPtr += matrix.power( 2, matrix.logWarpSize - group ) * stripLength;
					}
					elementPtr++;
				}
			}
			if( rowLengths.getElement( row ) > rowLength )
				ok = false;
		}
		if( ok )
			cout << "row lengths OK" << endl;
	}

	template< typename Real,
			  typename Index,
			  int StripSize >
	static void verifyRowPerm( const tnlBiEllpackMatrix< Real, Device, Index, StripSize >& matrix,
							   const typename tnlBiEllpackMatrix< Real, Device, Index, StripSize >::RowLengthsVector& rowLengths )
	{
		bool ok = true;
		Index numberOfStrips = matrix.virtualRows / matrix.warpSize;
		for( Index strip = 0; strip < numberOfStrips; strip++ )
		{
			Index begin = strip * matrix.warpSize;
			Index end = ( strip + 1 ) * matrix.warpSize;
			if( matrix.getRows() < end )
				end = matrix.getRows();
			for( Index i = begin; i < end - 1; i++ )
			{
				Index permIndex1, permIndex2;
				bool first = false;
				bool second = false;
				for( Index j = begin; j < end; j++ )
				{
					if( matrix.rowPermArray.getElement( j ) == i )
					{
						permIndex1 = j;
						first = true;
					}
					if( matrix.rowPermArray.getElement( j ) == i + 1 )
					{
						permIndex2 = j;
						second = true;
					}
				}
				if( !first || !second )
					cout << "nenasel jsem spravne indexy" << endl;
				if( rowLengths.getElement( permIndex1 ) >= rowLengths.getElement( permIndex2 ) )
					continue;
				else
					ok = false;
			}
		}
		if( ok )
			cout << "perm OK" << endl;
	}

	template< typename Real,
			  typename Index,
			  int StripSize >
	static void performRowBubbleSort( tnlBiEllpackMatrix< Real, Device, Index, StripSize >& matrix,
									  const typename tnlBiEllpackMatrix< Real, Device, Index, StripSize >::RowLengthsVector& rowLengths )
	{
#ifdef HAVE_CUDA
		Index numberOfStrips = matrix.virtualRows / StripSize;
		typedef tnlBiEllpackMatrix< Real, tnlCuda, Index, StripSize > Matrix;
		typedef typename Matrix::RowLengthsVector RowLengthsVector;
		Matrix* kernel_this = tnlCuda::passToDevice( matrix );
		RowLengthsVector* kernel_rowLengths = tnlCuda::passToDevice( rowLengths );
		dim3 cudaBlockSize( 256 ), cudaGridSize( tnlCuda::getMaxGridSize() );
		const Index cudaBlocks = roundUpDivision( numberOfStrips, cudaBlockSize.x );
		const Index cudaGrids = roundUpDivision( cudaBlocks, tnlCuda::getMaxGridSize() );
		for( int gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
		{
		     if( gridIdx == cudaGrids - 1 )
		         cudaGridSize.x = cudaBlocks % tnlCuda::getMaxGridSize();
		     performRowBubbleSortCuda< Real, Index, StripSize >
		     	 	 	 	 	 	 <<< cudaGridSize, cudaBlockSize >>>
		                             ( kernel_this,
		                               kernel_rowLengths,
		                               gridIdx );
		}
		tnlCuda::freeFromDevice( kernel_this );
		tnlCuda::freeFromDevice( kernel_rowLengths );
		checkCudaDevice;
#endif
	}

	template< typename Real,
			  typename Index,
			  int StripSize >
	static void computeColumnSizes( tnlBiEllpackMatrix< Real, Device, Index, StripSize >& matrix,
			 	 	 	 	 	 	const typename tnlBiEllpackMatrix< Real, Device, Index, StripSize >::RowLengthsVector& rowLengths )
	{
#ifdef HAVE_CUDA
		const Index numberOfStrips = matrix.virtualRows / StripSize;
		typedef tnlBiEllpackMatrix< Real, tnlCuda, Index, StripSize > Matrix;
		typedef typename Matrix::RowLengthsVector RowLengthsVector;
		Matrix* kernel_this = tnlCuda::passToDevice( matrix );
		RowLengthsVector* kernel_rowLengths = tnlCuda::passToDevice( rowLengths );
		dim3 cudaBlockSize( 256 ), cudaGridSize( tnlCuda::getMaxGridSize() );
		const Index cudaBlocks = roundUpDivision( numberOfStrips, cudaBlockSize.x );
		const Index cudaGrids = roundUpDivision( cudaBlocks, tnlCuda::getMaxGridSize() );
		for( int gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
		{
		     if( gridIdx == cudaGrids - 1 )
		         cudaGridSize.x = cudaBlocks % tnlCuda::getMaxGridSize();
		     computeColumnSizesCuda< Real, Index, StripSize >
		     	 	 	 	 	   <<< cudaGridSize, cudaBlockSize >>>
		                           ( kernel_this,
		                             kernel_rowLengths,
		                             numberOfStrips,
		                             gridIdx );
        }
		tnlCuda::freeFromDevice( kernel_this );
		tnlCuda::freeFromDevice( kernel_rowLengths );
		checkCudaDevice;
#endif
	}


	template< typename Real,
			  typename Index,
			  int StripSize,
			  typename InVector,
			  typename OutVector >
	static void vectorProduct( const tnlBiEllpackMatrix< Real, Device, Index, StripSize >& matrix,
			   	   	   	   	   const InVector& inVector,
			   	   	   	   	   OutVector& outVector )
	{
#ifdef HAVE_CUDA
		typedef tnlBiEllpackMatrix< Real, tnlCuda, Index > Matrix;
		Matrix* kernel_this = tnlCuda::passToDevice( matrix );
		InVector* kernel_inVector = tnlCuda::passToDevice( inVector );
		OutVector* kernel_outVector = tnlCuda::passToDevice( outVector );
		dim3 cudaBlockSize( 256 ), cudaGridSize( tnlCuda::getMaxGridSize() );
		const Index cudaBlocks = roundUpDivision( matrix.getRows(), cudaBlockSize.x );
		const Index cudaGrids = roundUpDivision( cudaBlocks, tnlCuda::getMaxGridSize() );
		for( Index gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
		{
			if( gridIdx == cudaGrids - 1 )
				cudaGridSize.x = cudaBlocks % tnlCuda::getMaxGridSize();
			const int sharedMemory = cudaBlockSize.x * sizeof( Real );
			tnlBiEllpackMatrixVectorProductCuda< Real, Index, StripSize, InVector, OutVector >
			                                   <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
			                                   ( kernel_this,
			                                     kernel_inVector,
			                                     kernel_outVector,
			                                     gridIdx,
			                                     matrix.warpSize );
		}
		tnlCuda::freeFromDevice( kernel_this );
		tnlCuda::freeFromDevice( kernel_inVector );
		tnlCuda::freeFromDevice( kernel_outVector );
		checkCudaDevice;
#endif
	}

};

#endif
