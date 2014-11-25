/* u addElement by nejspis melo byt realokovani pole jinak se asi prvek, ktery
 * na danem miste nebyl pridat neda, leda by se puvodne naalokovalo pole o neco vetsi
 *
 * u getRowLengths dat jeden cyklus co projede vsechny prvky a nastavi rovnou cele pole
 */
#ifndef TNLCOOMATRIX_H_
#define TNLCOOMATRIX_H_

#include <matrices/tnlSparseMatrix.h>
#include <core/vectors/tnlVector.h>

template< typename Device >
class tnlCOOMatrixDeviceDependentCode;

template< typename Real, typename Device = tnlHost, typename Index = int >
class tnlCOOMatrix : public tnlSparseMatrix < Real, Device, Index >
{
public:

	typedef Real RealType;
	typedef Device DeviceType;
	typedef Index IndexType;
	typedef typename tnlSparseMatrix< RealType, DeviceType, IndexType >:: RowLengthsVector RowLengthsVector;
	typedef tnlCOOMatrix< Real, Device, Index > ThisType;
	typedef tnlCOOMatrix< Real, tnlHost, Index > HostType;
	typedef tnlCOOMatrix< Real, tnlCuda, Index > CudaType;

	tnlCOOMatrix();

	static tnlString getType();

	tnlString getTypeVirtual() const;

	bool setDimensions(const IndexType rows,
			   	   	   const IndexType columns);

	void setNumberOfUsedValues();

	IndexType getNumberOfUsedValues() const;

	bool setRowLengths(const RowLengthsVector& rowLengths);

	void getRowLengths(tnlVector< IndexType, DeviceType, IndexType >& rowLengths) const;

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

	bool save(tnlFile& file) const;

	bool load(tnlFile& file);

	bool save(const tnlString& fileName) const;

	bool load(const tnlString& fileName);

	// nejsem si jisty jestli dela to co ma
	void print(ostream& str) const;

	void reset();

	typedef tnlCOOMatrixDeviceDependentCode< DeviceType > DeviceDependentCode;
	friend class tnlCOOMatrixDeviceDependentCode< DeviceType >;

private:

	tnlVector< Index, Device, Index > rowIndexes;
	
	IndexType numberOfUsedValues;

	int cudaWarpSize;

	bool appendMode;
};

#include <implementation/matrices/tnlCOOMatrix_impl.h>

#endif
