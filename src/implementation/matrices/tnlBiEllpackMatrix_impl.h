#ifndef TNLBIELLPACKMATRIX_IMPL_H_
#define TNLBIELLPACKMATRIX_IMPL_H_

#include <matrices/tnlBiEllpackMatrix.h>
#include <core/vectors/tnlVector.h>
#include <core/vectors/tnlSharedVector.h>
#include <core/mfuncs.h>

template< typename Real,
		  typename Device,
		  typename Index >
tnlBiEllpackMatrix< Real, Device, Index >::tnlBiEllpackMatrix()
: warpSize( 32 ),
  logWarpSize( 5 )
{}

template< typename Real,
		  typename Device,
		  typename Index >
tnlString tnlBiEllpackMatrix< Real, Device, Index >::getType()
{
	return tnlString( "BiEllpackMatrix< ") +
	       tnlString( ::getType< Real >() ) +
	       tnlString( ", " ) +
	       Device :: getDeviceType() +
	       tnlString( " >" );
}

template< typename Real,
		  typename Device,
		  typename Index >
tnlString tnlBiEllpackMatrix< Real, Device, Index >::getTypeVirtual() const
{
	return this->getType();
}

template< typename Real,
		  typename Device,
		  typename Index >
bool tnlBiEllpackMatrix< Real, Device, Index >::setDimensions( const IndexType rows,
															   const IndexType columns )
{
	tnlAssert( rows >= 0 && columns >= 0,
			   cerr << "rows = " << rows
			   	    << "columns = " <<columns <<endl );
	this->rows = rows;
	this->columns = columns;
	// dodelat
	return true;
}

template< typename Real,
		  typename Device,
		  typename Index >
bool tnlBiEllpackMatrix< Real, Device, Index >::setRowLengths(const RowLengthsVector& rowLengths)
{
	if( this->getRows() % this->warpSize != 0 )
		this->setVirtualRows( this->getRows() + this->getWarpSize() - ( this->getRows() % this->getWarpSize() ) );
	else
		this->setVirtualRows( this->getRows() );
	IndexType strips = this->getVirtualRows() / this->getWarpSize();

	if( !this->rowPermArray.setSize( this->getRows() ) ||
		!this->groupPointers.setSize( strips * ( this->logWarpSize + 1 ) + 1 )	)
		return false;

	for( IndexType row = 0; row < this->getRows(); row++ )
		this->rowPermArray.setElement(row, row);

	DeviceDependentCode::performRowBubbleSort( *this, rowLengths );

	DeviceDependentCode::computeColumnSizes( *this, rowLengths );

	this->groupPointers.computeExclusivePrefixSum();

	return
		this->allocateMatrixElements( this->getWarpSize() * this->groupPointers.getElement( strips * ( this->logWarpSize + 1 ) ) );
}

template< typename Real,
		  typename Device,
		  typename Index >
Index tnlBiEllpackMatrix< Real, Device, Index >::getNumberOfGroups( const IndexType row ) const
{
	IndexType strip = ( IndexType ) row / this->warpSize;
	IndexType rowStripPermutation = this->rowPermArray.getElement( row ) - this->warpSize * strip;
	IndexType numberOfGroups = this->logWarpSize + 1;
	while( rowStripPermutation > ( IndexType ) this->warpSize / pow( 2, numberOfGroups ) )
			numberOfGroups--;
	return numberOfGroups;
}

template< typename Real,
		  typename Device,
		  typename Index >
Index tnlBiEllpackMatrix< Real, Device, Index >::getRowLength( const IndexType row ) const
{
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
		  typename Index >
void tnlBiEllpackMatrix< Real, Device, Index >::getRowLengths( tnlVector< IndexType, DeviceType, IndexType >& rowLengths) const
{
	for( IndexType row; row < this->getRows(); row++ )
		rowLengths.setElement( row, this->getRowLength( row ) );
}

template< typename Real,
		  typename Device,
		  typename Index >
bool tnlBiEllpackMatrix< Real, Device, Index >::setElement( const IndexType row,
															const IndexType column,
															const RealType& value )
{
	return this->addElement( row, column, value, 0.0 );
}

template< typename Real,
		  typename Device,
		  typename Index >
bool tnlBiEllpackMatrix< Real, Device, Index >::addElement( const IndexType row,
															const IndexType column,
															const RealType& value,
															const RealType& thisElementMultiplicator )
{
	IndexType strip = ( IndexType ) row / this->warpSize;
	IndexType groupBegin = strip * ( this->logWarpSize + 1 );
	IndexType rowStripPermutation = this->rowPermArray.getElement( row ) - this->warpSize * strip;
	RealType result = 0.0;
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
		  typename Index >
bool tnlBiEllpackMatrix< Real, Device, Index >::setRow( const IndexType row,
														const IndexType* columns,
														const RealType* values,
														const IndexType numberOfElements )
{
	bool padding = false;
	IndexType strip = row / this->warpSize;
	IndexType groupBegin = strip * ( this->logWarpSize + 1 );
	IndexType rowStripPermutation = this->rowPermArray.getElement( row ) - this->warpSize * strip;
	IndexType elementPtr = 0;
	IndexType length = numberOfElements;
	IndexType i = 0;
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
		  typename Index >
bool tnlBiEllpackMatrix< Real, Device, Index >::addRow( const IndexType row,
														const IndexType* columns,
														const RealType* values,
														const IndexType numberOfElements,
														const RealType& thisElementMultiplicator )
{
	bool adding = true;
	IndexType strip = ( IndexType ) row / this->warpSize;
	IndexType groupBegin = strip * ( this->logWarpSize + 1 );
	IndexType rowStripPermutation = this->rowPermArray.getElement( row ) - this->warpSize * strip;
	RealType result = 0.0;
	for( IndexType elementPtr = 0; elementPtr < numberOfElements && adding; elementPtr++ )
	{
		bool add = false;
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
		  typename Index >
Real tnlBiEllpackMatrix< Real, Device, Index >::getElement( const IndexType row,
															const IndexType column ) const
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
				RealType value = this->values.getElement( this->warpSize * begin + rowStripPermutation + j * jumps );
				return value;
			}
		}
	}
	return 0.0;
}

template< typename Real,
		  typename Device,
		  typename Index >
void tnlBiEllpackMatrix< Real, Device, Index >::getRow( const IndexType row,
														IndexType* columns,
														RealType* values ) const
{
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
		  typename Index >
Index tnlBiEllpackMatrix< Real, Device, Index >::getWarpSize()
{
	return this->warpSize;
}

template< typename Real,
		  typename Device,
		  typename Index >
Index tnlBiEllpackMatrix< Real, Device, Index >::getVirtualRows()
{
	return this->virtualRows;
}

template< typename Real,
		  typename Device,
		  typename Index >
void tnlBiEllpackMatrix< Real, Device, Index >::setVirtualRows(const IndexType rows)
{
	this->virtualRows = rows;
}

template< typename Real,
		  typename Device,
		  typename Index >
Index tnlBiEllpackMatrix< Real, Device, Index >::getGroupLength( const Index strip,
																 const Index group ) const
{
	return this->groupPointers.getElement( strip * ( this->logWarpSize + 1 ) + group + 1 )
			- this->groupPointers.getElement( strip * ( this->warpSize + 1 ) + group );
}

template< typename Real,
	  	  typename Device,
	  	  typename Index >
template< typename InVector,
	  	  typename OutVector >
void tnlBiEllpackMatrix< Real, Device, Index >::vectorProduct( const InVector& inVector,
										  	  	  	  		   OutVector& outVector ) const
{
	DeviceDependentCode::vectorProduct( *this, inVector, outVector );
}

template< typename Real,
		  typename Device,
		  typename Index >
template< typename InVector >
typename InVector::RealType tnlBiEllpackMatrix< Real, Device, Index >::rowVectorProduct( const IndexType row,
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
		  typename Index >
void tnlBiEllpackMatrix< Real, Device, Index >::reset()
{
	tnlSparseMatrix< Real, Device, Index >::reset();
	this->rowPermArray.reset();
	this->groupPointers.reset();
}

template< typename Real,
		  typename Device,
		  typename Index >
bool tnlBiEllpackMatrix< Real, Device, Index >::save( tnlFile& file ) const
{

}

template< typename Real,
		  typename Device,
		  typename Index >
bool tnlBiEllpackMatrix< Real, Device, Index >::load( tnlFile& file )
{

}

template< typename Real,
		  typename Device,
		  typename Index >
bool tnlBiEllpackMatrix< Real, Device, Index >::save( const tnlString& fileName ) const
{

}

template< typename Real,
		  typename Device,
		  typename Index >
bool tnlBiEllpackMatrix< Real, Device, Index >::load( const tnlString& fileName )
{

}

template< typename Real,
		  typename Device,
		  typename Index >
void tnlBiEllpackMatrix< Real, Device, Index >::print( ostream& str ) const
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
			  typename InVector,
			  typename OutVector >
	static void vectorProduct( const tnlBiEllpackMatrix< Real, Device, Index >& matrix,
							   const InVector& inVector,
						       OutVector& outVector )
	{
		for( Index row = 0; row < matrix.getRows(); row++ )
			outVector[ row ] = matrix.rowVectorProduct( row, inVector );
	}

	template< typename Real,
			  typename Index >
	static void computeColumnSizes( tnlBiEllpackMatrix< Real, Device, Index >& matrix,
			 	 	 	 	 	 	const typename tnlBiEllpackMatrix< Real, Device, Index >::RowLengthsVector& rowLengths )
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
			  typename Index >
	static void performRowBubbleSort( tnlBiEllpackMatrix< Real, Device, Index >& matrix,
							   	   	  const typename tnlBiEllpackMatrix< Real, Device, Index >::RowLengthsVector& rowLengths)
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
		  typename Real >
__global__ void performRowBubbleSortCuda( tnlBiEllpackMatrix< Real, tnlCuda, Index >* matrix,
	 	 	  	  	  	  	   	   	   	  const typename tnlBiEllpackMatrix< Real, Device, Index >::RowLengthsVector* rowLengths )
{
	int i = threadIdx.x;
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
#endif

#ifdef HAVE_CUDA
template< typename Index,
		  typename Real >
void computeColumnSizesCuda( tnlBiEllpackMatrix< Index, tnlCuda, Real >* matrix,
							 const typename tnlBiEllpackMatrix< Real, Device, Index >::RowLengthsVector* rowLengths )
{
	int strip = threadIdx.x;
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
#endif

#ifdef HAVE_CUDA
template< typename Real,
		  typename Device,
		  typename Index >
template< typename InVector,
		  typename OutVector,
		  int warpSize >
__device__
void tnlBiEllpackMatrix< Real, tnlCude, Index >::spmvCuda( const InVector& inVector,
														   OutVector& outVector,
														   const IndexType warpStart,
														   const IndexType inWarpIdx ) const
{
	IndexType strip = warpStart / this->warpSize;
	IndexType groupBegin = ( warpStart / this->warpSize ) * ( this->logWarpSize + 1 );
	IndexType row = warpStart + inWarpIdx;
	IndexType rowStripPermutation = this->rowPermArray.getElement( row );
	Real* temp = getSharedMemory< Real >();
	temp[ threadIdx.x ] = 0.0;
	for( IndexType group; group < this->logWarpSize + 1; group++ )
	{
		if( rowStripPermutation % this->warpSize >= pow( 2, this->logWarpSize - group ) )
			rowStripPermutation = this->rowPermArray( row - pow( 2, this->logWarpSize ) );
		IndexType begin = this->groupPointers.getElement( groupBegin + group );
		for( IndexType j = 0; j < this->getGroupLength( strip, group ); j++ )
		{
			IndexType column = this->columnIndexes.getElement( this->warpSize * ( begin + j ) + rowStripPermutation % this->warpSize );
			if( column == this->getPaddingIndex() )
				continue;
			RealType value = this->values.getElement( this->warpSize * ( begin + j ) + rowStripPermutation % this->warpSize );
			temp[ rowStripPermutation ] += value * inVector[ column ];
		}
	}
	__syncthread();
}
#endif

#ifdef HAVE_CUDA
template< typename Real,
		  typename Index
		  typename InVector,
		  typename OutVector,
		  int warpSize >
__global__
void tnlBiEllpackMatrixVectorProductCudaKernel( const tnlBiEllpackMatrix< Real, tnlCuda, Index >* matrix,
												const InVector* inVector,
												OutVector* outVector,
												int gridIdx )
{
	IndexType globalIdx = ( gridIdx * tnlCuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
	const IndexType warpStart = warpSize * ( globalIdx / warpSize );
	const IndexType inWarpIdx = globalIdx % warpSize;
	matrix->spmvCuda( inVector, outVector, warpStart, inWarpIdx );
}
#endif

template<>
class tnlBiEllpackMatrixDeviceDependentCode< tnlCuda >
{
public:

	typedef tnlCuda Device;

	template< typename Index,
			  typename Real >
	static void performRowBubbleSort( tnlBiEllpackMatrix< Real, Device, Index >& matrix,
			 	 	 	 	 	 	  const typename tnlBiEllpackMatrix< Real, Device, Index >::RowLengthsVector& rowLengths )
	{
		typedef tnlBiEllpackMatrix< Real, tnlCuda, Index > Matrix;
		typedef typename Matrix::IndexType IndexType;
		Matrix* kernel_this = tnlCuda::passToDevice( matrix );
		RowLengthsVector* kernel_rowLengths = tnlCuda::passToDevice( rowLengths );
		const Index strips = matrix.roundUpDivision( matrix.getRows(), matrix.getWarpSize() );
		performRowBubbleSortCuda< Index, Real >
								<<< 1, strips >>>
								( kernel_this, kernel_rowLengths );
		tnlCuda::freeFromDevice( matrix );
		tnlCuda::freeFromDevice( kernel_rowLengths );
		checkCudaDevice;
	}

	template< typename Index,
			  typename Real >
	static void computeColumnSizes( tnlBiEllpackMatrix< Real, Device, Index >& matrix,
			 	 	 	 	 	 	const typename tnlBiEllpackMatrix< Real, Device, Index >::RowLengthsVector& rowLengths )
	{
		typedef tnlBiEllpackMatrix< Real, tnlCuda, Index > Matrix;
		typedef typename Matrix::IndexType IndexType;
		Matrix* kernel_this = tnlCuda::passToDevice( matrix );
		RowLengthsVector* kernel_rowLengths = tnlCuda::passToDevice( rowLengths );
		const Index strips = matrix.roundUpDivision( matrix.getRows(), matrix.getWarpSize() );
		computeColumnSizesCuda< Index, Real >
							  <<< 1, strips >>>
							  ( kernel_this, kernel_rowLengths );
		tnlCuda::freeFromDevice( matrix );
		tnlCuda::freeFromDevice( kernel_rowLengths );
		checkCudaDevice;
	}

	template< typename Index,
			  typename Real,
			  typename InVector,
			  typename OutVector >
	static void vectorProduct( const tnlBiEllpackMatrix< Real, Device, Index >& matrix,
			   	   	   	   	   const InVector& inVector,
			   	   	   	   	   OutVector& outVector )
	{
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
			tnlBiEllpackMatrixVectorProductCudaKernel< Real, Index, InVector, OutVector, matrix.warpSize >
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
	}
};

#endif
