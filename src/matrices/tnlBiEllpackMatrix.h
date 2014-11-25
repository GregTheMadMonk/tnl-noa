#ifndef TNLBIELLPACKMATRIX_H_
#define TNLBIELLPACKMATRIX_H_

template< typename Real, typename Device = tnlCuda, typename Index = int >
class tnlBiEllpackMatrix : public tnlSparseMatrix< Real, Device, Index >
{
public:
	typedef Real RealType;
	typedef Device DeviceType;
	typedef Index IndexType;
	typedef typename tnlSparseMatrix< RealType, DeviceType, IndexType >::RowLengthsVector RowLengthsVector;
	typedef typename tnlSparseMatrix< RealType, DeviceType, IndexType >::ValuesVector ValuesVector;
	typedef typename tnlSparseMatrix< RealType, DeviceType, IndexType >::ColumnIndexesVector ColumnIndexesVector;
	typedef tnlBiEllpackMatrix< Real, Device, Index > thisType;
	typedef tnlBiEllpackMatrix< Real, Device, Index > hostType;
	typedef tnlBiEllpackMatrix< Real, tnlCuda, Index > cudaType;

	static tnlString getType();

	tnlString getTypeVirtual() const;

	bool setDimensions( const IndexType rows,
						const IndexType columns );

	bool setRowLengths( const RowLengthsVector& rowLengths );

	void getRowLengths( tnlVector< IndexType, DeviceType, IndexType >& rowLengths ) const;

	IndexType getRowLength( const IndexType row ) const;

	bool setElement( const IndexType row,
					 const IndexType column,
					 const RealType& value );

	bool addElement( const IndexType row,
					 const IndexType column,
					 const RealType& value,
					 const thisElementMultiplicator& = 1.0 );

	bool setRow( const IndexType row,
				 const IndexType* columns,
				 const RealType* values,
				 const IndexType numberOfElements );

	bool addRow( const IndexType row,
				 const IndexType* columns,
				 const RealType* values,
				 const IndexType numberOfElements,
				 const RealType& thisElementMultiplicator = 1.0 );

	void performRowBubbleSort(const IndexType begin,
							  const IndexType end,
							  const RowLengthsVector& rowLengths);

	void sortCuda();

	void setVirtualRows(const IndexType rows);

	IndexType getVirtualRows();

	IndexType getWarpSize();




private:

	IndexType warpSize;

	IndexType virtualRows;

	tnlVector< Index, Device, Index > rowPermArray;

	tnlVector< Index, Device, Index > slicePointers;

	tnlVector< Index, Device, Index > sliceRowLengths;

	tnlVector< Index, Device, Index > rowLengths;

};


#endif
