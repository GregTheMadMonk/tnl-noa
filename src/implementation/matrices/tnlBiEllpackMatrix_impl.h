#ifndef TNLBIELLPACKMATRIX_IMPL_H_
#define TNLBIELLPACKMATRIX_IMPL_H_

#include <cmath>

template< typename Real,
		  typename Device,
		  typename Index >
tnlBiEllpackMatrix< Real, Device, Index >::tnlBiEllpackMatrix()
: warpSize( 32 )
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
	/* zjisti jestli je pocet radku delitelny 32 (velikosti warpu),
	 * pokud ne prida prazdne radky
 	 */
	IndexType remainder = this->getRows() % this->getWarpSize();
	if( remainder != 0 )
		this->setVirtualRows( this->getRows() + this->getWarpSize() - remainder );
	else
		this->setVirtualRows( this->getRows() );
	IndexType slices = this->getVirtualRows() / this->getWarpSize();


	if( !this->rowPermArray.setSize( this->getRows() ) ||
		!this->sliceRowLengths.setSize( slices ) ||
		!this->slicePointers.setSize( slices * 6 + 1 ) )
		return false;

	for( IndexType row = 0; row < this->getRows(); row++ )
		this->permArray.setElement(row, row);

	for( IndexType i = 0; i < slices; i++ )
		this->performRowBubbleSort( i * this->getWarpSize(), ( i + 1 ) * this->getWarpSize() - 1, rowLengths );

	DeviceDependentCode::computeColumnSizes( *this, rowLengths );


}

template< typename Real,
		  typename Device,
		  typename Index >
void tnlBiEllpackMatrix< Real, Device, Index >::getRowLengths( tnlVector< IndexType, Devicetype, Indextype >& rowLengths)
{

}

template< typename Real,
		  typename Device,
		  typename Index >
Index tnlBiEllpackMatrix< Real, Device, Index >::getRowLength( const IndexType row )
{

}

template< typename Real,
		  typename Device,
		  typename Index >
bool tnlBiEllpackMatrix< Real, Device, Index >::setElement( const IndexType row,
															const IndexType column,
															const RealType& value )
{

}

template< typename Real,
		  typename Device,
		  typename Index >
bool tnlBiEllpackMatrix< Real, Device, Index >::addElement( const IndexType row,
															const IndexType column,
															const RealType& value,
															const RealType& thisElementMultiplicator )
{

}

template< typename Real,
		  typename Device,
		  typename Index >
bool tnlBiEllpackMatrix< Real, Device, Index >::setRow( const IndexType row,
														const IndexType* columns,
														const RealType* values,
														const IndexType numberOfElements )
{

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
void tnlBiEllpackMatrix< Real, Device, Index >::performRowBubbleSort(const IndexType begin,
																	 const IndexType end,
																	 const RowLengthsVector& rowLengths)
{
	if(this->getRows() < end)
		end = this->getRows() - 1;
	bool sorted = false;
	IndexType offset = 0;
	while( !sorted )
	{
		sorted = true;
		for(IndexType i = begin + offset; i < end - offset; i++)
			if(rowLengths.getElement(this->rowPermArray.getElement(i)) < rowLengths.getElement(this->rowPermArray.getElement(i + 1)))
			{
				IndexType temp = this->rowPermArray.getElement(i);
				this->rowPermArray.setElement(i, this->rowPermArray.getElement(i + 1));
				this->rowPermArray.setElement(i + 1, temp);
				sorted = false;
			}
		for(IndexType i = end - 1 - offset; i > begin + offset; i--)
			if(rowLengths.getElement(this->rowPermArray.getElement(i)) > rowLengths.getElement(this->rowPermArray.getElement(i - 1)))
			{
				IndexType temp = this->rowPermArray.getElement(i);
				this->rowPermArray.setElement(i, this->rowPermArray.getElement(i - 1));
				this->rowPermArray.setElement(i - 1, temp);
				sorted = false;
			}
		offset++;
	}
}

template<>
class tnlBiEllpackMatrixDeviceDependentCode< tnlHost >
{
public:

	typedef tnlHost Device;

	template< typename Real,
			  typename Index,
			  int sliceSize >
	void computeColumnSizes( tnlBiEllpackMatrix< Real, Device, Index, SliceSize >& matrix,
			 	 	 	 	 const typename tnlSlicedEllpackMatrix< Real, Device, Index >::RowLengthsVector& rowLengths,
			 	 	 	 	 Index* groupArray )
	{
		Index slices = matrix.getVirtualRows() / matrix.getWarpSize();
		for( Index i = 0; i < slices; i++ )
			matrix.sliceRowLengths.setElement( i, this->computeColumnSize() );
	}

	template< typename Real,
			  typename Index,
			  int sliceSize >
	Index computeColumnSize( const Index begin,
							 const Index end,
			 	 	 	 	 const typename tnlSlicedEllpackMatrix< Real, Device, Index >::RowLengthsVector& rowLengths )
	{
		Index groupArray[ 6 ];
		Index numberOfGroups = 5;
		Index tempResult;
		for( Index group = 0; group < numberOfGroups; group++ )
		{
			tempResult = rowLengths.getElement( begin + pow(2, 4 - group ) );
			for( Index i = 0; i < group; i++ )
				tempResult -= ( Index ) pow( 2, i ) * groupArray[ i ];
			groupArray[ group ] = ( Index ) tempResult / pow( 2, group );
		}
		tempResult = rowLengths.getElement( begin );
		for( Index i = 0; i < numberOfGroups; i++ )
			tempResult -= ( Index ) pow( 2, i ) * groupArray[ i ];
		groupArray[ numberOfGroups ] = ( Index ) tempResult / pow( 2, numberOfGroups );
		Index length = 0;
		for( Index i = 0; i < 6; i++ )
			length += groupArray[ i ];
		return length;
	}
};


#endif
