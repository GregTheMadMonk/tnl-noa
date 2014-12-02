#ifndef TNLBIELLPACKMATRIX_IMPL_H_
#define TNLBIELLPACKMATRIX_IMPL_H_

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
	this->setVirtualRows( roundUpDivision( this->getRows(), this->getWarpSize() ) );
	IndexType slices = this->getVirtualRows() / this->getWarpSize();

	if( !this->rowPermArray.setSize( this->getRows() ) ||
		!this->groupPointers.setSize( slices * ( this->logWarpSize + 1 ) + 1 )	)
		return false;

	for( IndexType row = 0; row < this->getRows(); row++ )
		this->rowPermArray.setElement(row, row);

	DeviceDependentCode::performRowBubbleSort( *this, rowLengths );

	DeviceDependentCode::computeColumnSizes( *this, rowLengths );

	this->groupPointers.computeExclusivePrefixSum();

	return this->allocateMatrixElements( this->getWarpSize() * this->groupPointers.getElement( slices * ( this->logWarpSize + 1 ) ) );
}

template< typename Real,
		  typename Device,
		  typename Index >
Index tnlBiEllpackMatrix< Real, Device, Index >::getGroupLength( const Index strip,
																 const Index group ) const
{
	return this->groupPointers.getElement( strip * this->getWarpSize() + group + 1 )
			- this->groupPointers.getElement( strip * this->getWarpSize() + group );
}

template< typename Real,
		  typename Device,
		  typename Index >
void tnlBiEllpackMatrix< Real, Device, Index >::getRowLengths( tnlVector< IndexType, Devicetype, Indextype >& rowLengths)
{
	for( IndexType row; row < this->getRows(); row++ )
		this->getRowLength( row );
}

template< typename Real,
		  typename Device,
		  typename Index >
Index tnlBiEllpackMatrix< Real, Device, Index >::getRowLength( const IndexType row )
{
	return 0;
}

template< typename Real,
		  typename Device,
		  typename Index >
bool tnlBiEllpackMatrix< Real, Device, Index >::setElement( const IndexType row,
															const IndexType column,
															const RealType& value )
{
	return false;
}

template< typename Real,
		  typename Device,
		  typename Index >
bool tnlBiEllpackMatrix< Real, Device, Index >::addElement( const IndexType row,
															const IndexType column,
															const RealType& value,
															const RealType& thisElementMultiplicator )
{
	return false;
}

template< typename Real,
		  typename Device,
		  typename Index >
Real tnlBiEllpackMatrix< Real, Device, Index >::getElement( const IndexType row,
															const IndexType column ) const
{
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
	IndexType strip = row / this->getWarpSize();
	IndexType length = numberOfElements - this->getGroupLength( strip, i );
	IndexType i, elementPtr;
	i = elementPtr = 0;
	while( length >= 0 )
	{
		i++;
		length -= (IndexType) pow( 2, i ) * this->getGroupLength( strip, i );
	}
	length = numberOfElements;
	for( IndexType group = 0; group <= i; group++ )
	{
		IndexType rowBegin = this->getWarpSize() * this->groupPointers.getElement( ( this->logWarpSize + 1 ) * strip + group )
				+ this->rowPermArray.getElement( row );
		IndexType ratio = this->getWarpSize / pow( 2, group );
		for( IndexType j = 0; j < this->getGroupLength( strip, group ) && length != 0; j++ )
		{
			this->values.setElement( rowBegin + j * ratio, values[ elementPtr ]);
			this->columns.setElement( rowBegin + j * ratio, columns[ elementPtr ]);
			elementPtr++;
			length--;
		}
	}
}

template< typename Real,
		  typename Device,
		  typename Index >
template< typename InVector,
		  typename OutVector >
Real tnlBiEllpackMatrix< Real, Device, Index >::rowVectorProduct( const IndexType row,
																  const InVector& inVector )
{
	IndexType strip = row / this->getWarpSize();
	IndexType numberOfGroups = 6;
	RealType result = 0.0;
	while( row - strip * this->getWarpSize() > (IndexType) this->getWapSize() / pow( 2, numberOfGroups ) )
		numberOfGroups--;
	for( IndexType group = 0; group <= numberOfGroups; group++ )
	{
		IndexType rowBegin = this->getWarpSize() * this->groupPointers.getElement( ( this->logWarpSize + 1 ) * strip + group )
				+ this->rowPermArray.getElement( row );
		IndexType ratio = this->getWarpSize / pow( 2, group );
		for( IndexType j = 0; j < this->getGroupLength( strip, group ); j++ )
		{
			ReaType value = this->values.setElement( rowBegin + j * ratio, values[ elementPtr ]);
			IndexType column = this->columns.setElement( rowBegin + j * ratio, columns[ elementPtr ]);
			result += value * inVector[ column ];
		}
	}
	return result;
}

template< typename Real,
	  	  typename Device,
	  	  typename Index >
template< typename InVector,
	  	  typename OutVector >
void tnlBiEllpackMatrix< Real, Device, Index >::vectorProduct( const InVector& inVector,
										  	  	  	  		   OutVector& outVector )
{
	DeviceDependentCode::vectorProduct( *this, inVector, outVector );
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
	return false;
}

template< typename Real,
		  typename Device,
		  typename Index >
void tnlBiEllpackMatrix< Real, Device, Index >::getRow( const IndexType row,
														IndexType* columns,
														RealType* values ) const
{

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
void tnlBiEllpackMatrix< Real, Device, Index >::reset()
{
	tnlSparseMatrix< Real, Device, Index >::reset();
	this->rowPermArray.reset();
	this->groupPointers.reset();
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
	void vectorProduct( tnlBiEllpackMatrix< Real, Device, Index >& matrix,
						const InVector& inVector,
						OutVector& outVector )
	{
		for( Index row = 0; row < matrix.getRows(); row++ )
			outVector[ row ] = matrix.rowVectorProduct( row, inVector );
	}

	template< typename Real,
			  typename Index >
	void computeColumnSizes( tnlBiEllpackMatrix< Real, Device, Index >& matrix,
			 	 	 	 	 const typename tnlBiEllpackMatrix< Real, Device, Index >::RowLengthsVector& rowLengths )
	{
		Index numberOfStrips = matrix.getVirtualRows() / matrix.getWarpSize();
		for( Index strip = 0; strip < numberOfStrips - 1; strip++ )
			this->computeStripColumnSizes( strip, matrix, rowLengths );
		this->computeLastStripColumnSize( numberOfStrips - 1, matrix, rowLengths );

	}

	template< typename Real,
			  typename Index >
	void computeStripColumnSizes( const Index strip,
								  tnlBiEllpackMatrix< Real, Device, Index >& matrix,
								  const typename tnlBiEllpackMatrix< Real, Device, Index >::RowLengthsVector& rowLengths )
	{
		Index groupBegin = strip * ( matrix.logWarpSize + 1 );
		Index rowBegin = strip * matrix.getWarpSize();
		Index tempResult;
		for( Index group = groupBegin; group < groupBegin + matrix.logWarpSize; group++ )
		{
			tempResult = rowLengths.getElement( matrix.rowPermArray.getElement( rowBegin + pow(2, 4 - group + groupBegin ) ) );
			for( Index i = groupBegin; i < group + groupBegin; i++ )
				tempResult -= ( Index ) pow( 2, i ) * matrix.groupPointers.getElement( i );
			matrix.groupPointers.setElement( group, ceil( ( float ) tempResult / pow( 2, group - groupBegin ) ) );
		}
		tempResult = rowLengths.getElement( matrix.rowPermArray.getElement ( rowBegin ) );
		for( Index i = groupBegin; i < groupBegin + matrix.logWarpSize; i++ )
			tempResult -= ( Index ) pow( 2, i - groupBegin ) * matrix.groupPointers.getElement( i );
		matrix.groupPointers.setElement( groupBegin + matrix.logWarpSize, ceil( ( float ) tempResult / pow( 2, matrix.logWarpSize ) ) );
	}

	template< typename Real,
			  typename Index >
	void computeLastStripColumnSize( const Index lastStrip,
								     tnlBiEllpackMatrix< Real, Device, Index >& matrix,
				 	 	 	 	     const typename tnlBiEllpackMatrix< Real, Device, Index >::RowLengthsVector& rowLengths )
		{
			Index remaindingRows = matrix.getRows() - lastStrip * matrix.getWarpSize();
			Index i = 0;
			while( remaindingRows <= pow( 2, 5 - i ) )
				i++;
			Index groupBegin = lastStrip * ( matrix.logWarpSize + 1 );
			Index rowBegin = lastStrip * matrix.getWarpSize();
			for( Index j = groupBegin; j < groupBegin + i; j++ )
				matrix.groupPointer.setElement( j, 0 );

			for( Index group = groupBegin + i; group < groupBegin + matrix.logWarpSize; group++ )
			{
				tempResult = rowLengths.getElement( matrix.rowPermArray.getElement( rowBegin + pow(2, 4 - group + groupBegin ) ) );
				for( Index j = groupBegin; j < group + groupBegin; j++ )
					tempResult -= ( Index ) pow( 2, j ) * matrix.groupPointers.getElement( j );
				matrix.groupPointers.setElement( group, ceil( ( float ) tempResult / pow( 2, group - groupBegin ) ) );
			}
			tempResult = rowLengths.getElement( matrix.rowPermArray.getElement ( rowBegin ) );
			for( Index j = groupBegin; j < groupBegin + matrix.logWarpSize; j++ )
				tempResult -= ( Index ) pow( 2, j - groupBegin ) * matrix.groupPointers.getElement( j );
			matrix.groupPointers.setElement( groupBegin + matrix.logWarpSize, ceil( ( float ) tempResult / pow( 2, matrix.logWarpSize ) ) );
		}

	template< typename Real,
			  typename Device >
	void performRowBubbleSort( tnlBiEllpackMatrix< Real, Device, Index, SliceSize >& matrix,
							   const typename tnlBiEllpackMatrix< Real, Device, Index >::RowLengthsVector& rowLengths)
	{
		slices = matrix.getVirtualRows() / matrix.getWarpSize();
		for( IndexType i = 0; i < slices; i++ )
		{
			begin = i * matrix.getWarpSize();
			end = ( i + 1 ) * matrix.getWarpSize() - 1;
			if(matrix.getRows() < end)
				end = matrix.getRows() - 1;
			bool sorted = false;
			IndexType offset = 0;
			while( !sorted )
			{
				sorted = true;
				for(IndexType i = begin + offset; i < end - offset; i++)
					if(rowLengths.getElement(matrix.rowPermArray.getElement(i)) < rowLengths.getElement(matrix.rowPermArray.getElement(i + 1)))
					{
						IndexType temp = matrix.rowPermArray.getElement(i);
						matrix.rowPermArray.setElement(i, matrix.rowPermArray.getElement(i + 1));
						matrix.rowPermArray.setElement(i + 1, temp);
						sorted = false;
					}
				for(IndexType i = end - 1 - offset; i > begin + offset; i--)
					if(rowLengths.getElement(matrix.rowPermArray.getElement(i)) > rowLengths.getElement(matrix.rowPermArray.getElement(i - 1)))
					{
						IndexType temp = matrix.rowPermArray.getElement(i);
						matrix.rowPermArray.setElement(i, matrix.rowPermArray.getElement(i - 1));
						matrix.rowPermArray.setElement(i - 1, temp);
						sorted = false;
					}
				offset++;
			}
		}
	}
};


#endif
