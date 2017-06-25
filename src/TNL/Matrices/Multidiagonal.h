/***************************************************************************
                          Multidiagonal.h  -  description
                             -------------------
    begin                : Oct 13, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/Matrix.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/MultidiagonalRow.h>

namespace TNL {
namespace Matrices {   

template< typename Device >
class MultidiagonalDeviceDependentCode;

template< typename Real, typename Device = Devices::Host, typename Index = int >
class Multidiagonal : public Matrix< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename Matrix< Real, Device, Index >::CompressedRowLengthsVector CompressedRowLengthsVector;
   typedef Multidiagonal< Real, Device, Index > ThisType;
   typedef Multidiagonal< Real, Devices::Host, Index > HostType;
   typedef Multidiagonal< Real, Devices::Cuda, Index > CudaType;
   typedef Matrix< Real, Device, Index > BaseType;
   typedef MultidiagonalRow< Real, Index > MatrixRow;


   Multidiagonal();

   static String getType();

   String getTypeVirtual() const;

   static String getSerializationType();

   virtual String getSerializationTypeVirtual() const;

   void setDimensions( const IndexType rows,
                       const IndexType columns );

   void setCompressedRowLengths( const CompressedRowLengthsVector& rowLengths );

   IndexType getRowLength( const IndexType row ) const;

   __cuda_callable__
   IndexType getRowLengthFast( const IndexType row ) const;

   IndexType getMaxRowLength() const;

   template< typename Vector >
   void setDiagonals( const Vector& diagonals );

   const Containers::Vector< Index, Device, Index >& getDiagonals() const;

   template< typename Real2, typename Device2, typename Index2 >
   void setLike( const Multidiagonal< Real2, Device2, Index2 >& matrix );

   IndexType getNumberOfMatrixElements() const;

   IndexType getNumberOfNonzeroMatrixElements() const;

   IndexType getMaxRowlength() const;

   void reset();

   template< typename Real2, typename Device2, typename Index2 >
   bool operator == ( const Multidiagonal< Real2, Device2, Index2 >& matrix ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool operator != ( const Multidiagonal< Real2, Device2, Index2 >& matrix ) const;

   void setValue( const RealType& v );

   __cuda_callable__
   bool setElementFast( const IndexType row,
                        const IndexType column,
                        const RealType& value );

   bool setElement( const IndexType row,
                    const IndexType column,
                    const RealType& value );

   __cuda_callable__
   bool addElementFast( const IndexType row,
                        const IndexType column,
                        const RealType& value,
                        const RealType& thisElementMultiplicator = 1.0 );

   bool addElement( const IndexType row,
                    const IndexType column,
                    const RealType& value,
                    const RealType& thisElementMultiplicator = 1.0 );


   __cuda_callable__
   bool setRowFast( const IndexType row,
                    const IndexType* columns,
                    const RealType* values,
                    const IndexType numberOfElements );

   bool setRow( const IndexType row,
                const IndexType* columns,
                const RealType* values,
                const IndexType numberOfElements );


   __cuda_callable__
   bool addRowFast( const IndexType row,
                    const IndexType* columns,
                    const RealType* values,
                    const IndexType numberOfElements,
                    const RealType& thisElementMultiplicator = 1.0 );

   bool addRow( const IndexType row,
                const IndexType* columns,
                const RealType* values,
                const IndexType numberOfElements,
                const RealType& thisElementMultiplicator = 1.0 );

   __cuda_callable__
   RealType getElementFast( const IndexType row,
                            const IndexType column ) const;

   RealType getElement( const IndexType row,
                        const IndexType column ) const;

   __cuda_callable__
   void getRowFast( const IndexType row,
                    IndexType* columns,
                    RealType* values ) const;

   /*void getRow( const IndexType row,
                IndexType* columns,
                RealType* values ) const;*/

   __cuda_callable__
   MatrixRow getRow( const IndexType rowIndex );

   __cuda_callable__
   const MatrixRow getRow( const IndexType rowIndex ) const;

   template< typename Vector >
   __cuda_callable__
   typename Vector::RealType rowVectorProduct( const IndexType row,
                                               const Vector& vector ) const;

   template< typename InVector,
             typename OutVector >
   void vectorProduct( const InVector& inVector,
                       OutVector& outVector ) const;

   template< typename Real2, typename Index2 >
   void addMatrix( const Multidiagonal< Real2, Device, Index2 >& matrix,
                   const RealType& matrixMultiplicator = 1.0,
                   const RealType& thisMatrixMultiplicator = 1.0 );

   template< typename Real2, typename Index2 >
   void getTransposition( const Multidiagonal< Real2, Device, Index2 >& matrix,
                          const RealType& matrixMultiplicator = 1.0 );

   template< typename Vector >
   bool performSORIteration( const Vector& b,
                             const IndexType row,
                             Vector& x,
                             const RealType& omega = 1.0 ) const;

   bool save( File& file ) const;

   bool load( File& file );

   bool save( const String& fileName ) const;

   bool load( const String& fileName );

   void print( std::ostream& str ) const;

   protected:

   bool getElementIndex( const IndexType row,
                         const IndexType column,
                         IndexType& index ) const;

   __cuda_callable__
   bool getElementIndexFast( const IndexType row,
                             const IndexType column,
                             IndexType& index ) const;

   Containers::Vector< Real, Device, Index > values;

   Containers::Vector< Index, Device, Index > diagonalsShift;

   typedef MultidiagonalDeviceDependentCode< DeviceType > DeviceDependentCode;
   friend class MultidiagonalDeviceDependentCode< DeviceType >;


};

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/Multidiagonal_impl.h>
