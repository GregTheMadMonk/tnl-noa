/***************************************************************************
                          SparseMatrix.h -  description
                             -------------------
    begin                : Nov 29, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Matrices {

template< typename Real,
          typename Organization >
static String getSerializationType();

template< typename Real,
          typename Organization >
String getSerializationTypeVirtual() const;

template< typename Real,
          typename Organization >
void setDimensions( const IndexType rows,
                 const IndexType columns );

template< typename Real,
          typename Organization >
void setCompressedRowLengths( ConstCompressedRowLengthsVectorView rowLengths );

template< typename Real,
          typename Organization >
IndexType getRowLength( const IndexType row ) const;

template< typename Real,
          typename Organization >
__cuda_callable__
IndexType getRowLengthFast( const IndexType row ) const;

template< typename Real,
          typename Organization >
IndexType getNonZeroRowLength( const IndexType row ) const;

template< typename Real,
          typename Organization >
__cuda_callable__
IndexType getNonZeroRowLengthFast( const IndexType row ) const;

template< typename Real,
          typename Organization >
template< typename Real2, typename Device2, typename Index2 >
void setLike( const CSR< Real2, Device2, Index2 >& matrix );

template< typename Real,
          typename Organization >
void reset();

template< typename Real,
          typename Organization >
__cuda_callable__
bool setElementFast( const IndexType row,
                  const IndexType column,
                  const RealType& value );

template< typename Real,
          typename Organization >
bool setElement( const IndexType row,
              const IndexType column,
              const RealType& value );

template< typename Real,
          typename Organization >
__cuda_callable__
bool addElementFast( const IndexType row,
                  const IndexType column,
                  const RealType& value,
                  const RealType& thisElementMultiplicator = 1.0 );

template< typename Real,
          typename Organization >
bool addElement( const IndexType row,
              const IndexType column,
              const RealType& value,
              const RealType& thisElementMultiplicator = 1.0 );


template< typename Real,
          typename Organization >
__cuda_callable__
bool setRowFast( const IndexType row,
              const IndexType* columnIndexes,
              const RealType* values,
              const IndexType elements );

template< typename Real,
          typename Organization >
bool setRow( const IndexType row,
          const IndexType* columnIndexes,
          const RealType* values,
          const IndexType elements );


template< typename Real,
          typename Organization >
__cuda_callable__
bool addRowFast( const IndexType row,
              const IndexType* columns,
              const RealType* values,
              const IndexType numberOfElements,
              const RealType& thisElementMultiplicator = 1.0 );

template< typename Real,
          typename Organization >
bool addRow( const IndexType row,
          const IndexType* columns,
          const RealType* values,
          const IndexType numberOfElements,
          const RealType& thisElementMultiplicator = 1.0 );


template< typename Real,
          typename Organization >
__cuda_callable__
RealType getElementFast( const IndexType row,
                      const IndexType column ) const;

template< typename Real,
          typename Organization >
RealType getElement( const IndexType row,
                  const IndexType column ) const;

__cuda_callable__
template< typename Real,
          typename Organization >
void getRowFast( const IndexType row,
              IndexType* columns,
              RealType* values ) const;

template< typename Real,
          typename Organization >
__cuda_callable__
MatrixRow getRow( const IndexType rowIndex );

template< typename Real,
          typename Organization >
__cuda_callable__
ConstMatrixRow getRow( const IndexType rowIndex ) const;

template< typename Real,
          typename Organization >
template< typename Vector >
__cuda_callable__
typename Vector::RealType rowVectorProduct( const IndexType row,
                                         const Vector& vector ) const;

template< typename Real,
          typename Organization >
template< typename InVector,
       typename OutVector >
void vectorProduct( const InVector& inVector,
                 OutVector& outVector ) const;
// TODO: add const RealType& multiplicator = 1.0 )

template< typename Real,
          typename Organization >
template< typename Real2, typename Index2 >
void addMatrix( const CSR< Real2, Device, Index2 >& matrix,
             const RealType& matrixMultiplicator = 1.0,
             const RealType& thisMatrixMultiplicator = 1.0 );

template< typename Real,
          typename Organization >
template< typename Real2, typename Index2 >
void getTransposition( const CSR< Real2, Device, Index2 >& matrix,
                    const RealType& matrixMultiplicator = 1.0 );

template< typename Real,
          typename Organization >
template< typename Vector1, typename Vector2 >
bool performSORIteration( const Vector1& b,
                       const IndexType row,
                       Vector2& x,
                       const RealType& omega = 1.0 ) const;

// copy assignment
template< typename Real,
          typename Organization >
CSR& operator=( const CSR& matrix );

// cross-device copy assignment
template< typename Real,
          typename Organization >
template< typename Real2, typename Device2, typename Index2,
       typename = typename Enabler< Device2 >::type >
CSR& operator=( const CSR< Real2, Device2, Index2 >& matrix );

template< typename Real,
          typename Organization >
void save( File& file ) const;

template< typename Real,
          typename Organization >
void load( File& file );

template< typename Real,
          typename Organization >
void save( const String& fileName ) const;

template< typename Real,
          typename Organization >
void load( const String& fileName );

template< typename Real,
          typename Organization >
void print( std::ostream& str ) const;


   } //namespace Matrices
} // namespace  TNL
