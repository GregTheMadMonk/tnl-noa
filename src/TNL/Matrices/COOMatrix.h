/***************************************************************************
                          COOMatrix.h  -  description
                             -------------------
    begin                : Aug 27, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


/* u addElement by nejspis melo byt realokovani pole jinak se asi prvek, ktery
 * na danem miste nebyl pridat neda, leda by se puvodne naalokovalo pole o neco vetsi
 *
 * u getRowLengths dat jeden cyklus co projede vsechny prvky a nastavi rovnou cele pole
 */
#pragma once

#include <TNL/Matrices/Sparse.h>
#include <TNL/Containers/Vectors/Vector.h>

namespace TNL {
   namespace Matrices

template< typename Device >
class COOMatrixDeviceDependentCode;

template< typename Real, typename Device = Devices::Host, typename Index = int >
class COOMatrix : public Sparse < Real, Device, Index >
{
public:

	typedef Real RealType;
	typedef Device DeviceType;
	typedef Index IndexType;
	typedef typename Sparse< RealType, DeviceType, IndexType >:: CompressedRowLengthsVector CompressedRowLengthsVector;
	typedef COOMatrix< Real, Device, Index > ThisType;
	typedef COOMatrix< Real, Devices::Host, Index > HostType;
	typedef COOMatrix< Real, tnlCuda, Index > CudaType;

	COOMatrix();

	static String getType();

	String getTypeVirtual() const;

	bool setDimensions(const IndexType rows,
			   	   	   const IndexType columns);

	void setNumberOfUsedValues();

	IndexType getNumberOfUsedValues() const;

	bool setCompressedRowLengths(const CompressedRowLengthsVector& rowLengths);

	void getRowLengths(Containers::Vector< IndexType, DeviceType, IndexType >& rowLengths) const;

	IndexType getRowLength( const IndexType row ) const;

	bool setElement(const IndexType row,
					const IndexType column,
					const RealType& value);

	bool addElement(const IndexType row,
					const IndexType column,
					const RealType& value,
					const RealType& thisElementMultiplicator = 1.0);

	bool appendElement( const IndexType row,
						const IndexType column,
						const RealType& value);

	bool setRow(const IndexType row,
				const IndexType* columns,
				const RealType* values,
				const IndexType numberOfElements);

	bool addRow(const IndexType row,
				const IndexType* columns,
				const RealType* values,
				const IndexType numberOfElements,
				const RealType& thisElementMultiplicator = 1.0);

	Real getElement(const IndexType row,
					const IndexType column) const;

	void getRow(const IndexType row,
				IndexType* columns,
				RealType* values) const;

	template< typename InVector,
		  	  typename OutVector >
	void vectorProduct(const InVector& inVector,
			   	   	   OutVector& outVector) const;

	template< typename InVector,
			  typename OutVector >
	void vectorProductHost(const InVector& inVector,
						   OutVector& outVector) const;

	template< typename Vector >
	typename Vector::RealType rowVectorProduct(const IndexType row,
											   const Vector& inVector) const;

	bool save(File& file) const;

	bool load(File& file);

	bool save(const String& fileName) const;

	bool load(const String& fileName);

	// nejsem si jisty jestli dela to co ma
	void print(ostream& str) const;

	void reset();

	typedef COOMatrixDeviceDependentCode< DeviceType > DeviceDependentCode;
	friend class COOMatrixDeviceDependentCode< DeviceType >;

private:

	Containers::Vector< Index, Device, Index > rowIndexes;
	
	IndexType numberOfUsedValues;

	int cudaWarpSize;

	bool appendMode;
};

   } //namespace Matrices
} // namespace TNL


#include <TNL/Matrices/COOMatrix_impl.h>

#endif
