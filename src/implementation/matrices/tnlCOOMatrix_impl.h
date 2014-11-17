#ifndef TNLCOOMATRIX_IMPL_H_
#define TNLCOOMATRIX_IMPL_H_

#include <matrices/tnlCOOMatrix.h>
#include <core/vectors/tnlVector.h>
#include <core/vectors/tnlSharedVector.h>
#include <core/mfuncs.h>

template< typename Real,
	  	  typename Device,
	  	  typename Index >
tnlCOOMatrix< Real, Device, Index >::tnlCOOMatrix()
:cudaWarpSize( 32 ),
 numberOfUsedValues( 0 )
{
};

template< typename Real,
	  	  typename Device,
	  	  typename Index >
tnlString tnlCOOMatrix< Real, Device, Index >::getType()
{
	return tnlString("tnlCOOMatrix< ") +
  	 	   tnlString(::getType< Real>()) +
		   tnlString(", ") +
		   Device::getDeviceType() +
		   tnlString(" >");
}

template< typename Real,
	  	  typename Device,
	  	  typename Index >
tnlString tnlCOOMatrix< Real, Device, Index >::getTypeVirtual() const
{
	return this->getType();
}

template< typename Real,
		  typename Device,
		  typename Index >
bool tnlCOOMatrix< Real, Device, Index >::setDimensions(const IndexType rows, const IndexType columns)
{
	if (!tnlSparseMatrix< Real, Device, Index >::setDimensions(rows, columns) ||
	    !this->rowIndexes.setSize( this->values.getSize() ) )
		return false;
	return true;
}

template< typename Real,
	  	  typename Device,
	  	  typename Index >
void tnlCOOMatrix< Real, Device, Index >::setNumberOfUsedValues()
{
	for(IndexType i = 0; i < this->values.getSize(); i++)
		{
			if(this->values[ i ] == 0.0)
				{
					numberOfUsedValues = i;
					break;
				}
		}
}

template< typename Real,
	  	  typename Device,
	  	  typename Index >
Index tnlCOOMatrix< Real, Device, Index >::getNumberOfUsedValues() const
{
	return this->numberOfUsedValues;
}

template< typename Real,
		  typename Device,
		  typename Index >
bool tnlCOOMatrix< Real, Device, Index >::setRowLengths(const RowLengthsVector& rowLengths)
{
	IndexType size = 0;
	for(IndexType row = 0; row < this->getRows(); row++)
		size += rowLengths.getElement(row);
	if( !this->rowIndexes.setSize(size) ||
		!this->columnIndexes.setSize(size) ||
		!this->values.setSize(size) )
		return false;
	return true;
}

template< typename Real,
		  typename Device,
		  typename Index >
void tnlCOOMatrix< Real, Device, Index >::getRowLengths(tnlVector< IndexType, DeviceType, IndexType >& rowLengthsVector) const
{
	for(IndexType row = 0; row < this->getRows(); row++)
		rowLengthsVector.setElement(row, this->getRowLength(row));
}

template< typename Real,
		  typename Device,
		  typename Index >
Index tnlCOOMatrix< Real, Device, Index >::getRowLength( const IndexType row ) const
{
	IndexType rowLength = 0;
	for(IndexType elementPtr = 0; elementPtr < this->values.getSize(); elementPtr++)
		if(rowIndexes.getElement(elementPtr) == row)
			rowLength++;
	return rowLength;
}

template< typename Real,
		  typename Device,
		  typename Index >
bool tnlCOOMatrix< Real, Device, Index >::setElement( const IndexType row,
						      	  	  	  	  	  	  const IndexType column,
						      	  	  	  	  	  	  const RealType& value)
{
	return this->addElement( row, column, value, 1.0);
}

template< typename Real,
	  	  typename Device,
	  	  typename Index >
bool tnlCOOMatrix< Real, Device, Index >::addElement( const IndexType row, 
						      	  	  	  	  	  	  const IndexType column,
						      	  	  	  	  	  	  const RealType& value,
						      	  	  	  	  	  	  const RealType& thisElementMultiplicator )
{
	tnlAssert( row >= 0 && row < this->rows &&
               column >= 0 && column < this->columns,
               cerr << " row = " << row
                    << " column = " << column
                    << " this->rows = " << this->rows
                    << " this->columns = " << this->columns );

	IndexType endPtr = this->getNumberOfUsedValues();
	for(IndexType elementPtr = 0; elementPtr < endPtr; elementPtr++)
	{
		if(this->rowIndexes[ elementPtr ] == row && this->columnIndexes[ elementPtr ] == column)
		{
			this->values.setElement( elementPtr, thisElementMultiplicator * this->values.getElement( elementPtr ) + value );
			return true;
		}	
	}
	if(endPtr < this->values.getSize())
	{
		this->values.setElement( endPtr, thisElementMultiplicator * this->values.getElement( endPtr ) + value );
		this->rowIndexes.setElement( endPtr, row );
		this->columnIndexes.setElement( endPtr, column );
		this->numberOfUsedValues++;
		return true;
	}
	return false;
}

template< typename Real,
		  typename Device,
		  typename Index >
bool tnlCOOMatrix< Real, Device, Index >::setRow(const IndexType row,
												 const IndexType* columns,
												 const RealType* values,
												 const IndexType numberOfElements)
{
	IndexType rowPtr = getNumberOfUsedValues();
	IndexType end = rowPtr + numberOfElements;
	for(IndexType i = 0; i < numberOfElements; i++)
	{
		this->rowIndexes.setElement(rowPtr, row);
		this->columnIndexes.setElement(rowPtr, columns[i]);
		this->values.setElement(rowPtr, values[i]);
		rowPtr++;
	}
	numberOfUsedValues += numberOfElements;
	return true;
}

template< typename Real,
		  typename Device,
		  typename Index >
bool tnlCOOMatrix< Real, Device, Index >::addRow(const IndexType row,
												 const IndexType* columns,
												 const RealType* values,
												 const IndexType numberOfElements,
												 const RealType& thisElementMultiplicator )
{
	// ma secist dva radky? popr. jak?
	return false;
}

template< typename Real,
		  typename Device,
		  typename Index >
Real tnlCOOMatrix< Real, Device, Index >::getElement(const IndexType row,
													 const IndexType column) const
{
	for (IndexType elementPtr = 0; elementPtr < this->getNumberOfUsedValues(); elementPtr++)
	{
		if (this->rowIndexes[elementPtr] == row && this->columnIndexes[elementPtr] == column)
			return this->values.getElement(elementPtr);
	}
	return 0.0;
}

template< typename Real,
		  typename Device,
		  typename Index >
void tnlCOOMatrix< Real, Device, Index >::getRow(const IndexType row,
												 IndexType* columns,
												 RealType* values) const
{
	IndexType i = 0;
	for (IndexType elementPtr; elementPtr < this->getNumberOfUsedValues(); elementPtr++)
	{
		if (this->rowIndexes[elementPtr] == row)
		{
			columns[i] = this->columnIndexes.getElement(elementPtr);
			values[i] = this->values.getElement(elementPtr);
			i++;
		}
	}
}

template< typename Real,
	  	  typename Device,
	  	  typename Index >
template< typename InVector,
	  	  typename OutVector >
void tnlCOOMatrix< Real, Device, Index >::vectorProduct(const InVector& inVector,
										  	  	  	  	OutVector& outVector) const
{
	DeviceDependentCode::vectorProduct( *this, inVector, outVector );
}

template< typename Real,
		  typename Device,
		  typename Index >
template< typename InVector,
		  typename OutVector >
void tnlCOOMatrix< Real, Device, Index >::vectorProductHost(const InVector& inVector,
															OutVector& outVector) const
{
	for(IndexType i = 0; i < this->values.getSize(); i++)
		outVector[ this->rowIndexes.getElement(i) ] = this->values.getElement(i)*inVector[ this->columnIndexes.getElement(i) ];
}

template <>
class tnlCOOMatrixDeviceDependentCode< tnlHost >
{
	public:
		typedef tnlHost Device;

		template< typename Real,
			  	  typename Index,
			  	  typename InVector,
			  	  typename OutVector >
		static void vectorProduct( const tnlCOOMatrix< Real, Device, Index >& matrix,
						  	  	   const InVector& inVector,
						  	  	   OutVector& outVector)
		{
			matrix.vectorProductHost( inVector, outVector );
		}
};

template< typename Real,
		  typename Device,
		  typename Index >
template< typename Vector >
typename Vector::RealType tnlCOOMatrix< Real, Device, Index >::rowVectorProduct(const IndexType row,
																				const Vector& inVector) const
{
	RealType result = 0.0;
	for(IndexType elementPtr = 0; elementPtr < this->values.getSize(); elementPtr++)
		if(this->rowIndexes.getElement(elementPtr) == row)
		{
			result += this->values.getElement(elementPtr) * inVector[this->columnIndexes.getElement(elementPtr)];
		}
	return result;
}

template< typename Real,
	  	  typename Device,
	  	  typename Index >
bool tnlCOOMatrix< Real, Device, Index >::save(tnlFile& file) const
{
	if (!tnlSparseMatrix< Real, Device, Index >::save(file) ||
	    !this->rowIndexes.save(file))
		return false;
	return true;
}

template< typename Real,
		  typename Device,
		  typename Index >
bool tnlCOOMatrix< Real, Device, Index >::load(tnlFile& file)
{
	if (!tnlSparseMatrix< Real, Device, Index >::load(file) ||
	    !this->rowIndexes.load(file))
		return false;
	return true;
}

template< typename Real,
		  typename Device,
		  typename Index >
bool tnlCOOMatrix< Real, Device, Index >::save(const tnlString& fileName) const
{
	return tnlObject::save(fileName);
}

template< typename Real,
	  	  typename Device,
	  	  typename Index >
bool tnlCOOMatrix< Real, Device, Index >::load(const tnlString& fileName)
{
	return tnlObject::load(fileName);
}

template< typename Real,
		  typename Device,
		  typename Index >
void tnlCOOMatrix< Real, Device, Index >::print(ostream& str) const
{
	//zatim jsem to napsal takhle, kdyztak to pozdeji zmenim
	for(IndexType elementPtr = 0; elementPtr < this->getNumberOfUsedValues(); elementPtr++)
	{
		str << "Row: " << this->rowIndexes.getElement(elementPtr) << "\t";
		str << "Col: " << this->columnIndexes.getElement(elementPtr) << " -> ";
		str << this->values.getElement(elementPtr) << "\n";
	}
}

template< typename Real,
		  typename Device,
		  typename Index >
void tnlCOOMatrix< Real, Device, Index >::reset()
{
	tnlSparseMatrix< Real, Device, Index >::reset();
	this->rowIndexes.reset();
}

#endif
