/***************************************************************************
                          tnlMatrix.h  -  description
                             -------------------
    begin                : Dec 18, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Object.h>
#include <TNL/core/tnlHost.h>
#include <TNL/Vectors/Vector.h>

namespace TNL {

template< typename Real = double,
          typename Device = tnlHost,
          typename Index = int >
class tnlMatrix : public virtual Object
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef Vectors::tnlVector< IndexType, DeviceType, IndexType > CompressedRowsLengthsVector;
   typedef Vectors::tnlVector< RealType, DeviceType, IndexType > ValuesVector;

   tnlMatrix();

   virtual bool setDimensions( const IndexType rows,
                               const IndexType columns );

   virtual bool setCompressedRowsLengths( const CompressedRowsLengthsVector& rowLengths ) = 0;

   virtual IndexType getRowLength( const IndexType row ) const = 0;

   virtual void getCompressedRowsLengths( Vectors::tnlVector< IndexType, DeviceType, IndexType >& rowLengths ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool setLike( const tnlMatrix< Real2, Device2, Index2 >& matrix );

   virtual IndexType getNumberOfMatrixElements() const = 0;

   virtual IndexType getNumberOfNonzeroMatrixElements() const = 0;

   void reset();

   __cuda_callable__
   IndexType getRows() const;

   __cuda_callable__
   IndexType getColumns() const;

   /****
    * TODO: The fast variants of the following methods cannot be virtual.
    * If they were, they could not be used in the CUDA kernels. If CUDA allows it
    * in the future and it does not slow down, declare them as virtual here.
    */

   virtual bool setElement( const IndexType row,
                            const IndexType column,
                            const RealType& value ) = 0;

   virtual bool addElement( const IndexType row,
                            const IndexType column,
                            const RealType& value,
                            const RealType& thisElementMultiplicator = 1.0 ) = 0;

   virtual bool setRow( const IndexType row,
                        const IndexType* columns,
                        const RealType* values,
                        const IndexType numberOfElements ) = 0;

   virtual bool addRow( const IndexType row,
                        const IndexType* columns,
                        const RealType* values,
                        const IndexType numberOfElements,
                        const RealType& thisElementMultiplicator = 1.0 ) = 0;

   virtual Real getElement( const IndexType row,
                            const IndexType column ) const = 0;

   tnlMatrix< RealType, DeviceType, IndexType >& operator = ( const tnlMatrix< RealType, DeviceType, IndexType >& );

   template< typename Matrix >
   bool operator == ( const Matrix& matrix ) const;

   template< typename Matrix >
   bool operator != ( const Matrix& matrix ) const;

   template< typename Matrix >
   bool copyFrom( const Matrix& matrix,
                  const CompressedRowsLengthsVector& rowLengths );

   virtual bool save( File& file ) const;

   virtual bool load( File& file );

   virtual void print( std::ostream& str ) const;

   protected:

   IndexType rows, columns;

   public: // TODO: remove this

   ValuesVector values;
};

template< typename Real, typename Device, typename Index >
std::ostream& operator << ( std::ostream& str, const tnlMatrix< Real, Device, Index >& m )
{
   m.print( str );
   return str;
}

template< typename Matrix,
          typename InVector,
          typename OutVector >
void tnlMatrixVectorProductCuda( const Matrix& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector );

} // namespace TNL

#include <TNL/matrices/tnlMatrix_impl.h>
