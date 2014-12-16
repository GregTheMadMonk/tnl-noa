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

	cout << "setDimensions" << endl;

	if( this->getRows() % this->warpSize != 0 )
		this->setVirtualRows( this->getRows() + this->getWarpSize() - ( this->getRows() % this->getWarpSize() ) );
	else
		this->setVirtualRows( this->getRows() );
	IndexType strips = this->getVirtualRows() / this->getWarpSize();

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
	cout << "setRowLengths" << endl;
	if( this->getRows() % this->warpSize != 0 )
		this->setVirtualRows( this->getRows() + this->getWarpSize() - ( this->getRows() % this->getWarpSize() ) );
	else
		this->setVirtualRows( this->getRows() );
	IndexType strips = this->getVirtualRows() / this->getWarpSize();
	if( ! this->rowPermArray.setSize( this->rows ) ||
		! this->groupPointers.setSize( strips * ( this->logWarpSize + 1 ) + 1 ) )
		return false;

	DeviceDependentCode::performRowBubbleSort( *this, rowLengths );

	DeviceDependentCode::computeColumnSizes( *this, rowLengths );

	this->groupPointers.computeExclusivePrefixSum();

	return
		this->allocateMatrixElements( this->getWarpSize() * this->groupPointers.getElement( strips * ( this->logWarpSize + 1 ) ) );
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
Index tnlBiEllpackMatrix< Real, Device, Index, StripSize >::getNumberOfGroups( const IndexType row ) const
{
	tnlAssert( row >=0 && row < this->getRows(),
	              cerr << "row = " << row
	                   << " this->getRows() = " << this->getRows()
	                   << " this->getName() = " << this->getName() << endl );

	IndexType strip = ( IndexType ) row / this->warpSize;
	IndexType rowStripPermutation = this->rowPermArray.getElement( row ) - this->warpSize * strip;
	IndexType numberOfGroups = this->logWarpSize + 1;
	while( rowStripPermutation > ( IndexType ) this->warpSize / pow( 2, numberOfGroups ) )
			numberOfGroups--;
	return numberOfGroups;
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
Index tnlBiEllpackMatrix< Real, Device, Index, StripSize >::getRowLength( const IndexType row ) const
{
	tnlAssert( row >=0 && row < this->getRows(),
	              cerr << "row = " << row
	                   << " this->getRows() = " << this->getRows()
	                   << " this->getName() = " << this->getName() << endl );

	IndexType strip = ( IndexType ) row / this->warpSize;
	IndexType groupBegin = strip * ( this->logWarpSize + 1 );
	IndexType rowStripPermutation = this->rowPermArray.getElement( row ) - this->warpSize * strip;
	IndexType length = 0;
	for( IndexType group = 0; group < this->getNumberOfGroups( row ); group++ )
	{
		IndexType begin = this->groupPointers.getElement( groupBegin + group );
		IndexType end = this->groupPointers.getElement( groupBegin + group + 1 );
		IndexType jumps = this->warpSize / pow( 2, group );
		for( IndexType j = 0; j < pow( 2, group ) * ( end - begin ); j++ )
		{
			RealType value = this->values.getElement( this->warpSize * begin + rowStripPermutation + j * jumps );
			if( value == 0.0 )
				return length;
			else
				length++;
		}
	}
	return length;
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
	IndexType strip = ( IndexType ) row / this->warpSize;
	IndexType groupBegin = strip * ( this->logWarpSize + 1 );
	IndexType rowStripPermutation = this->rowPermArray.getElement( row ) - this->warpSize * strip;
	for( IndexType group = 0; group < this->getNumberOfGroups( row ); group++ )
	{
		IndexType begin = this->groupPointers.getElement( groupBegin + group );
		IndexType end = this->groupPointers.getElement( groupBegin + group + 1 );
		IndexType jumps = this->warpSize / pow( 2, group );
		for( IndexType j = 0; j < pow( 2, group ) * ( end - begin ); j++ )
		{
			IndexType columnCheck = this->columnIndexes.getElement( this->warpSize * begin + rowStripPermutation + j * jumps );
			if( columnCheck == column )
			{
				RealType valueCheck = this->values.getElement( this->warpSize * begin + rowStripPermutation + j * jumps );
				RealType result = valueCheck + value * thisElementMultiplicator;
				this->values.setElement( this->warpSize * begin + rowStripPermutation + j * jumps, result );
				return true;
			}
			if( columnCheck == this->getPaddingIndex() )
			{
				this->values.setElement( this->warpSize * begin + rowStripPermutation + j * jumps, value );
				this->columnIndexes.setElement( this->warpSize * begin + rowStripPermutation + j * jumps, column );
				return true;
			}
		}
	}
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
	tnlAssert( row >=0 && row < this->getRows(),
		              cerr << "row = " << row
		                   << " this->getRows() = " << this->getRows()
		                   << " this->getName() = " << this->getName() << endl );

	IndexType strip = row / this->warpSize;
	IndexType groupBegin = strip * ( this->logWarpSize + 1 );
	IndexType rowStripPermutation = this->rowPermArray.getElement( row ) - this->warpSize * strip;
	IndexType elementPtr = 0;
	for( IndexType group = 0; ( group < this->getNumberOfGroups( row ) ) && ( elementPtr < numberOfElements ); group++ )
	{
		IndexType j;
		IndexType begin = this->groupPointers.getElement( groupBegin + group );
		IndexType end = this->groupPointers.getElement( groupBegin + group + 1 );
		IndexType jumps = this->warpSize / pow( 2, group );
		for( j = 0; ( j < pow( 2, group ) * ( end - begin ) ) && ( elementPtr < numberOfElements ); j++ )
		{
			this->columnIndexes.setElement( this->warpSize * begin + rowStripPermutation + j * jumps, columns[ elementPtr ] );
			this->values.setElement( this->warpSize * begin + rowStripPermutation + j * jumps, values[ elementPtr ] );
			elementPtr++;
		}
		if( elementPtr == numberOfElements - 1 )
			for( IndexType i = j; i < pow( 2, group ) * ( end - begin ); i++ )
				this->columnIndexes.setElement( this->warpSize * begin + rowStripPermutation + i * jumps, this->getPaddingIndex() );
	}
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
	IndexType strip = ( IndexType ) row / this->warpSize;
	IndexType groupBegin = strip * ( this->logWarpSize + 1 );
	IndexType rowStripPermutation = this->rowPermArray.getElement( row ) - this->warpSize * strip;
	for( IndexType elementPtr = 0; elementPtr < numberOfElements && adding; elementPtr++ )
	{
		for( IndexType group = 0; group < this->getNumberOfGroups( row ) && adding; group++ )
		{
			adding = false;
			IndexType begin = this->groupPointers.getElement( groupBegin + group );
			IndexType end = this->groupPointers.getElement( groupBegin + group + 1 );
			IndexType jumps = this->warpSize / pow( 2, group );
			for( IndexType j = 0; j < pow( 2, group ) * ( end - begin ) && adding; j++ )
			{
				IndexType columnCheck = this->columnIndexes.getElement( this->warpSize * begin + rowStripPermutation + j * jumps );
				if( columnCheck == columns[ elementPtr ] )
				{
					RealType valueCheck = this->values.getElement( this->warpSize * begin + rowStripPermutation + j * jumps );
					RealType result = valueCheck + values[ elementPtr ] * thisElementMultiplicator;
					this->values.setElement( this->warpSize * begin + rowStripPermutation + j * jumps, result );
					adding = true;
				}
				if( columnCheck == this->getPaddingIndex() )
				{
					this->columnIndexes.setElement( this->warpSize * begin + rowStripPermutation + j * jumps, columns[ elementPtr ] );
					this->values.setElement( this->warpSize * begin + rowStripPermutation + j * jumps, values[ elementPtr ] );
					adding = true;
				}
			}
		}
	}
	return adding;
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
Real tnlBiEllpackMatrix< Real, Device, Index, StripSize >::getElement( const IndexType row,
																	   const IndexType column ) const
{
	tnlAssert( ( row >=0 && row < this->getRows() ) ||
			    ( column >= 0 && column < this->getColumns() ),
	              cerr << "row = " << row
	                   << " this->getRows() = " << this->getRows()
	                   << " this->getColumns() = " << this->getColumns()
	                   << " this->getName() = " << this->getName() << endl );

	IndexType strip = ( IndexType ) row / this->warpSize;
	IndexType groupBegin = strip * ( this->logWarpSize + 1 );
	IndexType rowStripPermutation = this->rowPermArray.getElement( row ) - this->warpSize * strip;
	for( IndexType group = 0; group < this->getNumberOfGroups( row ); group++ )
	{
		IndexType begin = this->groupPointers.getElement( groupBegin + group );
		IndexType end = this->groupPointers.getElement( groupBegin + group + 1 );
		IndexType jumps = this->warpSize / pow( 2, group );
		for( IndexType j = 0; j < pow( 2, group ) * ( end - begin ); j++ )
		{
			IndexType columnCheck = this->columnIndexes.getElement( this->warpSize * begin + rowStripPermutation + j * jumps );
			if( columnCheck == column )
			{
				RealType value = this->values.getElement( this->warpSize * begin + rowStripPermutation + j * jumps );
				return value;
			}
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
	IndexType strip = ( IndexType ) row / this->warpSize;
	IndexType groupBegin = strip * ( this->logWarpSize + 1 );
	IndexType rowStripPermutation = this->rowPermArray.getElement( row ) - this->warpSize * strip;
	IndexType elementPtr = 0;
	for( IndexType group = 0; group < this->getNumberOfGroups( row ) && !padding; group++ )
	{
		IndexType begin = this->groupPointers.getElement( groupBegin + group );
		IndexType end = this->groupPointers.getElement( groupBegin + group + 1 );
		IndexType jumps = this->warpSize / pow( 2, group );
		for( IndexType j = 0; j < pow( 2, group ) * ( end - begin ) && !padding; j++ )
		{
			IndexType column = this->columnIndexes.getElement( this->warpSize * begin + rowStripPermutation + j * jumps );
			if( column == this->getPaddingIndex() )
			{
				padding = true;
				break;
			}
			RealType value = this->values.getElement( this->warpSize * begin + rowStripPermutation + j * jumps );
			values[ elementPtr ] = value;
			columns[ elementPtr ] = column;
			elementPtr++;
		}
	}
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
Index tnlBiEllpackMatrix< Real, Device, Index, StripSize >::getWarpSize()
{
	return this->warpSize;
}

template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
Index tnlBiEllpackMatrix< Real, Device, Index, StripSize >::getVirtualRows()
{
	return this->virtualRows;
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
Index tnlBiEllpackMatrix< Real, Device, Index, StripSize >::getGroupLength( const Index strip,
																 	 	    const Index group ) const
{
	return this->groupPointers.getElement( strip * ( this->logWarpSize + 1 ) + group + 1 )
			- this->groupPointers.getElement( strip * ( this->warpSize + 1 ) + group );
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
template< typename InVector >
typename InVector::RealType tnlBiEllpackMatrix< Real, Device, Index, StripSize >::rowVectorProduct( const IndexType row,
																  	  	  	  	  	     	 	    const InVector& inVector ) const
{
	bool padding = false;
	IndexType strip = ( IndexType ) row / this->warpSize;
	IndexType groupBegin = strip * ( this->logWarpSize + 1 );
	IndexType rowStripPermutation = this->rowPermArray.getElement( row ) - this->warpSize * strip;
	RealType result = 0.0;
	for( IndexType group = 0; group < this->getNumberOfGroups( row ) && !padding; group++ )
	{
		IndexType begin = this->groupPointers.getElement( groupBegin + group );
		IndexType end = this->groupPointers.getElement( groupBegin + group + 1 );
		IndexType jumps = this->warpSize / pow( 2, group );
		for( IndexType j = 0; j < pow( 2, group ) * ( end - begin ) && !padding; j++ )
		{
			IndexType column = this->columnIndexes.getElement( this->warpSize * begin + rowStripPermutation + j * jumps );
			if( column == this->getPaddingIndex() )
			{
				padding = true;
				break;
			}
			RealType value = this->values.getElement( this->warpSize * begin + rowStripPermutation + j * jumps );
			result += value * inVector[ column ];
		}
	}
	return result;
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
		IndexType strip = ( IndexType ) row / this->warpSize;
		IndexType groupBegin = strip * ( this->logWarpSize + 1 );
		IndexType rowStripPermutation = this->rowPermArray.getElement( row ) - this->warpSize * strip;
		for( IndexType group = 0; group < this->getNumberOfGroups( row ) && !padding; group++ )
		{
			IndexType begin = this->groupPointers.getElement( groupBegin + group );
			IndexType end = this->groupPointers.getElement( groupBegin + group + 1 );
			IndexType jumps = this->warpSize / pow( 2, group );
			for( IndexType j = 0; j < pow( 2, group ) * ( end - begin ) && !padding; j++ )
			{
				IndexType column = this->columnIndexes.getElement( this->warpSize * begin + rowStripPermutation + j * jumps );
				if( column == this->getPaddingIndex() )
				{
					padding = true;
					break;
				}
				RealType value = this->values.getElement( this->warpSize * begin + rowStripPermutation + j * jumps );
				str << " Col:" << column << "->" << value << "\t";
			}
		}
		str << endl;
	}
}

template<>
class tnlBiEllpackMatrixDeviceDependentCode< tnlHost >
{
public:

	typedef tnlHost Device;

	template< typename Real,
			  typename Index,
			  int StripSize,
			  typename InVector,
			  typename OutVector >
	static void vectorProduct( const tnlBiEllpackMatrix< Real, Device, Index, StripSize >& matrix,
							   const InVector& inVector,
						       OutVector& outVector )
	{
		for( Index row = 0; row < matrix.getRows(); row++ )
			outVector[ row ] = matrix.rowVectorProduct( row, inVector );
	}

	template< typename Real,
			  typename Index,
			  int StripSize >
	static void computeColumnSizes( tnlBiEllpackMatrix< Real, Device, Index, StripSize >& matrix,
			 	 	 	 	 	 	const typename tnlBiEllpackMatrix< Real, Device, Index, StripSize >::RowLengthsVector& rowLengths )
	{
		Index numberOfStrips = matrix.getVirtualRows() / matrix.getWarpSize();
		for( Index strip = 0; strip < numberOfStrips; strip++ )
		{
			Index i = 0;
			Index rowBegin = strip * matrix.getWarpSize();
			Index groupBegin = strip * ( matrix.logWarpSize + 1 );
			Index emptyGroups = 0;
			if( strip == numberOfStrips - 1 )
			{
				while( matrix.getRows() < ( Index ) pow( 2, matrix.logWarpSize - 1 - emptyGroups ) + rowBegin + 1 )
					emptyGroups++;
				for( Index group = groupBegin; group < groupBegin + emptyGroups; group++ )
					matrix.groupPointers.setElement( group, 0 );
			}
			i += emptyGroups;
			for( Index group = groupBegin + emptyGroups; group < groupBegin + matrix.logWarpSize + 1; group++ )
			{
				Index row = ( Index ) rowBegin + pow( 2, 4 - i );
				Index temp = rowLengths.getElement( matrix.rowPermArray.getElement( row ) );
				for( Index prevGroups = groupBegin; prevGroups < group; prevGroups++ )
					temp -= pow( 2, prevGroups - groupBegin ) * matrix.groupPointers.getElement( prevGroups );
				temp =  ceil( ( float ) temp / pow( 2, i ) );
				matrix.groupPointers.setElement( group, temp );
				i++;
			}
		}
	}

	template< typename Real,
			  typename Index,
			  int StripSize >
	static void performRowBubbleSort( tnlBiEllpackMatrix< Real, Device, Index, StripSize >& matrix,
							   	   	  const typename tnlBiEllpackMatrix< Real, Device, Index, StripSize >::RowLengthsVector& rowLengths)
	{
		Index strips = matrix.getVirtualRows() / matrix.getWarpSize();
		for( Index i = 0; i < strips; i++ )
		{
			Index begin = i * matrix.getWarpSize();
			Index end = ( i + 1 ) * matrix.getWarpSize() - 1;
			if(matrix.getRows() - 1 < end)
				end = matrix.getRows() - 1;
			bool sorted = false;
			Index offset = 0;
			while( !sorted )
			{
				sorted = true;
				for(Index i = begin + offset; i < end - offset; i++)
					if(rowLengths.getElement(matrix.rowPermArray.getElement(i)) < rowLengths.getElement(matrix.rowPermArray.getElement(i + 1)))
					{
						Index temp = matrix.rowPermArray.getElement(i);
						matrix.rowPermArray.setElement(i, matrix.rowPermArray.getElement(i + 1));
						matrix.rowPermArray.setElement(i + 1, temp);
						sorted = false;
					}
				for(Index i = end - 1 - offset; i > begin + offset; i--)
					if(rowLengths.getElement(matrix.rowPermArray.getElement(i)) > rowLengths.getElement(matrix.rowPermArray.getElement(i - 1)))
					{
						Index temp = matrix.rowPermArray.getElement(i);
						matrix.rowPermArray.setElement(i, matrix.rowPermArray.getElement(i - 1));
						matrix.rowPermArray.setElement(i - 1, temp);
						sorted = false;
					}
				offset++;
			}
		}
	}
};

#ifdef HAVE_CUDA
template< typename Index,
		  typename Real,
		  int StripSize >
__global__ void performRowBubbleSortCudaKernel( tnlBiEllpackMatrix< Real, tnlCuda, Index, StripSize >* matrix,
	 	 	  	  	  	  	   	   	   	  	    const typename tnlBiEllpackMatrix< Real, tnlCuda, Index, StripSize >::RowLengthsVector* rowLengths )
{
	int i = threadIdx.x;
	Index begin = i * matrix->warpSize;
	Index end = ( i + 1 ) * matrix->warpSize - 1;
	if(matrix->getRows() - 1 < end)
		end = matrix->getRows() - 1;
	bool sorted = false;
	Index offset = 0;
	while( !sorted )
	{
		sorted = true;
		for(Index i = begin + offset; i < end - offset; i++)
		if(rowLengths->getElement(matrix->rowPermArray.getElement(i)) < rowLengths->getElement(matrix->rowPermArray.getElement(i + 1)))
		{
			Index temp = matrix->rowPermArray.getElement(i);
			matrix->rowPermArray.setElement(i, matrix->rowPermArray.getElement(i + 1));
			matrix->rowPermArray.setElement(i + 1, temp);
			sorted = false;
		}
		for(Index i = end - 1 - offset; i > begin + offset; i--)
		if(rowLengths->getElement(matrix->rowPermArray.getElement(i)) > rowLengths->getElement(matrix->rowPermArray.getElement(i - 1)))
		{
			Index temp = matrix->rowPermArray.getElement(i);
			matrix->rowPermArray.setElement(i, matrix->rowPermArray.getElement(i - 1));
			matrix->rowPermArray.setElement(i - 1, temp);
			sorted = false;
		}
		offset++;
	}
}
#endif

#ifdef HAVE_CUDA
template< typename Index,
		  typename Real,
		  int StripSize >
void computeColumnSizesCudaKernel( tnlBiEllpackMatrix< Index, tnlCuda, Real, StripSize >* matrix,
							       const typename tnlBiEllpackMatrix< Real, tnlCuda, Index, StripSize >::RowLengthsVector* rowLengths,
							       const Index numberOfStrips )
{
	int strip = threadIdx.x;
	Index i = 0;
	Index rowBegin = strip * matrix->getWarpSize();
	Index groupBegin = strip * ( matrix->logWarpSize + 1 );
	Index emptyGroups = 0;
	if( strip == numberOfStrips - 1 )
	{
		while( matrix->getRows() < ( Index ) pow( 2, matrix->logWarpSize - 1 - emptyGroups ) + rowBegin + 1 )
			emptyGroups++;
		for( Index group = groupBegin; group < groupBegin + emptyGroups; group++ )
			matrix->groupPointers.setElement( group, 0 );
	}
	i += emptyGroups;
	for( Index group = groupBegin + emptyGroups; group < groupBegin + matrix->logWarpSize + 1; group++ )
	{
		Index row = ( Index ) rowBegin + pow( 2, 4 - i );
		Index temp = rowLengths->getElement( matrix->rowPermArray.getElement( row ) );
		for( Index prevGroups = groupBegin; prevGroups < group; prevGroups++ )
		temp -= pow( 2, prevGroups - groupBegin ) * matrix->groupPointers.getElement( prevGroups );
		temp =  ceil( ( float ) temp / pow( 2, i ) );
		matrix->groupPointers.setElement( group, temp );
		i++;
	}
}
#endif

#ifdef HAVE_CUDA
template< typename Real,
		  typename Device,
		  typename Index,
		  int StripSize >
template< typename InVector,
		  typename OutVector,
		  int warpSize >
__device__
void tnlBiEllpackMatrix< Real, Device, Index, StripSize >::spmvCuda( const InVector& inVector,
														  	  	  	 OutVector& outVector,
														  	  	  	 const IndexType warpStart,
														  	  	  	 const IndexType inWarpIdx ) const
{
	IndexType strip = warpStart / this->warpSize;
	IndexType groupBegin = strip * ( this->logWarpSize + 1 );
	IndexType row = warpStart + inWarpIdx;
	IndexType rowStripPermutation = this->rowPermArray.getElement( row );
	IndexType step = row % this->warpSize;
	Real* temp = getSharedMemory< Real >();
	temp[ threadIdx.x ] = 0.0;
	for( IndexType group; group < this->logWarpSize + 1; group++ )
	{
		if( rowStripPermutation % this->warpSize >= pow( 2, this->logWarpSize - group ) )
			rowStripPermutation = this->rowPermArray.getElement( row - pow( 2, this->logWarpSize ) );
		IndexType begin = this->groupPointers.getElement( groupBegin + group );
		for( IndexType j = 0; j < this->getGroupLength( strip, group ); j++ )
		{
			IndexType column = this->columnIndexes.getElement( this->warpSize * ( begin + j ) + step );
			if( column == this->getPaddingIndex() )
				continue;
			RealType value = this->values.getElement( this->warpSize * ( begin + j ) + step );
			temp[ rowStripPermutation ] += value * inVector[ column ];
		}
	}
	__syncthreads();
}
#endif

#ifdef HAVE_CUDA
template< typename Real,
		  typename Index,
		  int StripSize,
		  typename InVector,
		  typename OutVector,
		  int warpSize >
__global__
void tnlBiEllpackMatrixVectorProductCudaKernel( const tnlBiEllpackMatrix< Real, tnlCuda, Index, StripSize >* matrix,
												const InVector* inVector,
												OutVector* outVector,
												int gridIdx )
{
	Index globalIdx = ( gridIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
	const Index warpStart = warpSize * ( globalIdx / warpSize );
	const Index inWarpIdx = globalIdx % warpSize;
	matrix->spmvCuda( inVector, outVector, warpStart, inWarpIdx );
}
#endif

template< typename Real,
		  typename Index,
		  int StripSize,
		  typename InVector,
		  typename OutVector >
void tnlBiEllpackMatrixVectorProductCuda( const tnlBiEllpackMatrix< Real, tnlCuda, Index, StripSize >& matrix,
										  const InVector& inVector,
										  OutVector& outVector )
{
#ifdef HAVE_CUDA
	typedef tnlBiEllpackMatrix< Real, tnlCuda, Index > Matrix;
	typedef typename Matrix::IndexType IndexType;
	Matrix* kernel_this = tnlCuda::passToDevice( matrix );
	InVector* kernel_inVector = tnlCuda::passToDevice( inVector );
	OutVector* kernel_outVector = tnlCuda::passToDevice( outVector );
	dim3 cudaBlockSize( 256 ), cudaGridSize( tnlCuda::getMaxGridSize() );
	const IndexType cudaBlocks = roundUpDivision( matrix.getRows(), cudaBlockSize.x );
	const IndexType cudaGrids = roundUpDivision( cudaBlocks, tnlCuda::getMaxGridSize() );
	for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
	{
		if( gridIdx == cudaGrids - 1 )
			cudaGridSize.x = cudaBlocks % tnlCuda::getMaxGridSize();
		const int sharedMemory = cudaBlockSize.x * sizeof( Real );
		tnlBiEllpackMatrixVectorProductCudaKernel< Real, Index, StripSize, InVector, OutVector, 32 >
		                                  	     <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
		                                   	     ( kernel_this,
		                                   	       kernel_inVector,
		                                   	       kernel_outVector,
		                                   	       gridIdx );
	}
	freeFromDevice( matrix );
	freeFromDevice( inVector );
	freeFromDevice( outVector );
	checkCudaDevice;
#endif
}

#ifdef HAVE_CUDA
template< typename Real,
		  typename Index,
		  int StripSize,
		  typename RowLengthsVector >
void computecolumnSizesCuda( const tnlBiEllpackMatrix< Real, tnlCuda, Index, StripSize >& matrix,
							 const RowLengthsVector& rowLengths )
{
	typedef tnlBiEllpackMatrix< Real, tnlCuda, Index > Matrix;
	typedef typename Matrix::IndexType IndexType;
	Matrix* kernel_this = tnlCuda::passToDevice( matrix );
	RowLengthsVector* kernel_rowLengths = tnlCuda::passToDevice( rowLengths );
	const Index strips = matrix.roundUpDivision( matrix.getRows(), matrix.getWarpSize() );
	computeColumnSizesCudaKernel< Real, Index, StripSize >
						  	    <<< 1, strips >>>
						  	    ( kernel_this, kernel_rowLengths, strips );
	tnlCuda::freeFromDevice( kernel_this );
	tnlCuda::freeFromDevice( kernel_rowLengths );
	checkCudaDevice;
}
#endif

#ifdef HAVE_CUDA
template< typename Real,
		  typename Index,
		  int StripSize,
		  typename RowLengthsVector >
void performRowBubbleSortCuda( const tnlBiEllpackMatrix< Real, tnlCuda, Index, StripSize >& matrix,
							   const RowLengthsVector& rowLengths )
{
	typedef tnlBiEllpackMatrix< Real, tnlCuda, Index > Matrix;
	typedef typename Matrix::IndexType IndexType;
	Matrix* kernel_this = tnlCuda::passToDevice( matrix );
	RowLengthsVector* kernel_rowLengths = tnlCuda::passToDevice( rowLengths );
	const Index strips = matrix.roundUpDivision( matrix.getRows(), matrix.getWarpSize() );
	performRowBubbleSortCudaKernel< Real, Index, StripSize >
								  <<< 1, strips >>>
								  ( kernel_this, kernel_rowLengths );
	tnlCuda::freeFromDevice( matrix );
	tnlCuda::freeFromDevice( kernel_rowLengths );
	checkCudaDevice;
}
#endif

template<>
class tnlBiEllpackMatrixDeviceDependentCode< tnlCuda >
{
public:

	typedef tnlCuda Device;

	template< typename Index,
			  typename Real,
			  int StripSize >
	static void performRowBubbleSort( tnlBiEllpackMatrix< Real, Device, Index, StripSize >& matrix,
			 	 	 	 	 	 	  const typename tnlBiEllpackMatrix< Real, Device, Index, StripSize >::RowLengthsVector& rowLengths )
	{
		performRowBubbleSortCuda( matrix, rowLengths );
	}

	template< typename Index,
			  typename Real,
			  int StripSize >
	static void computeColumnSizes( tnlBiEllpackMatrix< Real, Device, Index, StripSize >& matrix,
			 	 	 	 	 	 	const typename tnlBiEllpackMatrix< Real, Device, Index, StripSize >::RowLengthsVector& rowLengths )
	{
		computeColumnSizesCuda( matrix, rowLengths );
	}

	template< typename Index,
			  typename Real,
			  int StripSize,
			  typename InVector,
			  typename OutVector >
	static void vectorProduct( const tnlBiEllpackMatrix< Real, Device, Index, StripSize >& matrix,
			   	   	   	   	   const InVector& inVector,
			   	   	   	   	   OutVector& outVector )
	{
		tnlBiEllpackMatrixVectorProductCuda( matrix, inVector, outVector );
	}
};

#endif
