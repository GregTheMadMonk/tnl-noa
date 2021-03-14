/***************************************************************************
                          COOMatrix.h  -  description
                             -------------------
    begin                : Aug 27, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/COOMatrix.h>
#include <TNL/Math.h>
#include <TNL/TypeInfo.h>

namespace TNL {
namespace Matrices {

template< typename Real,
	  	  typename Device,
	  	  typename Index >
COOMatrix< Real, Device, Index >::COOMatrix()
:cudaWarpSize( 32 ),
 numberOfUsedValues( 0 ),
 appendMode( true )
{
};

template< typename Real,
	  	  typename Device,
	  	  typename Index >
String COOMatrix< Real, Device, Index >::getType()
{
	return String( "Matrices::COOMatrix< " ) +
               String( TNL::getType< Real>() ) +
               String( ", " ) +
               String( Device :: getDeviceType() ) +
               String( ", " ) +
               String( TNL::getType< Index >() ) +
               String( " >" );
}

template< typename Real,
	  	  typename Device,
	  	  typename Index >
String COOMatrix< Real, Device, Index >::getTypeVirtual() const
{
	return this->getType();
}

template< typename Real,
		  typename Device,
		  typename Index >
bool COOMatrix< Real, Device, Index >::setDimensions(const IndexType rows, const IndexType columns)
{
	if (!Sparse< Real, Device, Index >::setDimensions(rows, columns) ||
	    !this->rowIndexes.setSize( this->values.getSize() ) )
		return false;
	return true;
}

template< typename Real,
	  	  typename Device,
	  	  typename Index >
void COOMatrix< Real, Device, Index >::setNumberOfUsedValues()
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
Index COOMatrix< Real, Device, Index >::getNumberOfUsedValues() const
{
	return this->numberOfUsedValues;
}

template< typename Real,
		  typename Device,
		  typename Index >
bool COOMatrix< Real, Device, Index >::setCompressedRowLengths(ConstRowsCapacitiesTypeView rowLengths)
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
void COOMatrix< Real, Device, Index >::getRowLengths(Containers::Vector< IndexType, DeviceType, IndexType >& rowLengthsVector) const
{
	IndexType rowLength;
	for(IndexType row = 0; row < this->getRows(); row++)
		rowLengthsVector.setElement(row, 0);
	for(IndexType elementPtr = 0; elementPtr < this->values.getSize(); elementPtr++)
	{
		rowLength = rowLengthsVector.getElement(this->rowIndexes.getElement(elementPtr));
		rowLengthsVector.setElement(this->rowIndexes.getElement(elementPtr), rowLength++);
	}
}

template< typename Real,
		  typename Device,
		  typename Index >
Index COOMatrix< Real, Device, Index >::getRowLength( const IndexType row ) const
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
bool COOMatrix< Real, Device, Index >::setElement( const IndexType row,
						      	  	  	  	  	  	  const IndexType column,
						      	  	  	  	  	  	  const RealType& value )
{
	if( this->appendMode )
		return this->addElement( row, column, value, 1.0 );
	else
		return this->appendElement( row, column, value );
}

template< typename Real,
	  	  typename Device,
	  	  typename Index >
bool COOMatrix< Real, Device, Index >::addElement( const IndexType row, 
						      	  	  	  	  	  	  const IndexType column,
						      	  	  	  	  	  	  const RealType& value,
						      	  	  	  	  	  	  const RealType& thisElementMultiplicator )
{
	TNL_ASSERT( row >= 0 && row < this->rows &&
               column >= 0 && column < this->columns,
              std::cerr << " row = " << row
                    << " column = " << column
                    << " this->rows = " << this->rows
                    << " this->columns = " << this->columns );
	if( appendMode )
		return this->appendElement( row, column, value );

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
bool COOMatrix< Real, Device, Index >::appendElement( const IndexType row,
														 const IndexType column,
														 const RealType& value )
{
	if( !this->getNumberOfUsedValues < this->values.getSize() )
		return false;
	else
	{
		this->rowIndexes.setElement( this->getNumberOfUsedValues(), row );
		this->columnIndexes.setElement( this->getNumberOfUsedValues(), column );
		this->values.setElement( this->getNumberOfUsedValues(), value );
		this->numberOfUsedValues++;
	}
	return true;
}

template< typename Real,
		  typename Device,
		  typename Index >
bool COOMatrix< Real, Device, Index >::setRow(const IndexType row,
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
bool COOMatrix< Real, Device, Index >::addRow(const IndexType row,
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
Real COOMatrix< Real, Device, Index >::getElement(const IndexType row,
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
void COOMatrix< Real, Device, Index >::getRow(const IndexType row,
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
void COOMatrix< Real, Device, Index >::vectorProduct(const InVector& inVector,
										  	  	  	  	OutVector& outVector) const
{
	DeviceDependentCode::vectorProduct( *this, inVector, outVector );
}

template< typename Real,
		  typename Device,
		  typename Index >
template< typename InVector,
		  typename OutVector >
void COOMatrix< Real, Device, Index >::vectorProductHost(const InVector& inVector,
															OutVector& outVector) const
{
	for(IndexType row = 0; row < this->getRows(); row++)
		outVector[ row ] = 0;
	for(IndexType i = 0; i < this->values.getSize(); i++)
		outVector[ this->rowIndexes.getElement(i) ] = this->values.getElement(i)*inVector[ this->columnIndexes.getElement(i) ];
}

template <>
class COOMatrixDeviceDependentCode< Devices::Host >
{
	public:
		typedef Devices::Host Device;

		template< typename Real,
			  	  typename Index,
			  	  typename InVector,
			  	  typename OutVector >
		static void vectorProduct( const COOMatrix< Real, Device, Index >& matrix,
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
typename Vector::RealType COOMatrix< Real, Device, Index >::rowVectorProduct(const IndexType row,
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
void COOMatrix< Real, Device, Index >::save(File& file) const
{
	Sparse< Real, Device, Index >::save(file);
	file << this->rowIndexes;
}

template< typename Real,
		  typename Device,
		  typename Index >
void COOMatrix< Real, Device, Index >::load(File& file)
{
	Sparse< Real, Device, Index >::load(file);
	file >> this->rowIndexes;
}

template< typename Real,
		  typename Device,
		  typename Index >
void COOMatrix< Real, Device, Index >::save(const String& fileName) const
{
	Object::save(fileName);
}

template< typename Real,
	  	  typename Device,
	  	  typename Index >
void COOMatrix< Real, Device, Index >::load(const String& fileName)
{
	Object::load(fileName);
}

template< typename Real,
		  typename Device,
		  typename Index >
void COOMatrix< Real, Device, Index >::print(std::ostream& str) const
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
void COOMatrix< Real, Device, Index >::reset()
{
	Sparse< Real, Device, Index >::reset();
	this->rowIndexes.reset();
}

} // namespace Matrices
} // namespace TNL
