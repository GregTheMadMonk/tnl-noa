/***************************************************************************
                          Matrix.h  -  description
                             -------------------
    begin                : Dec 18, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Object.h>
#include <TNL/Allocators/Default.h>
#include <TNL/Devices/Host.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Matrices/MatrixView.h>

namespace TNL {
/**
 * \brief Namespace for matrix formats.
 */
namespace Matrices {

template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          typename RealAllocator = typename Allocators::Default< Device >::template Allocator< Real > >
class Matrix : public Object
{
public:
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using CompressedRowLengthsVector = Containers::Vector< IndexType, DeviceType, IndexType >;
   using CompressedRowLengthsVectorView = Containers::VectorView< IndexType, DeviceType, IndexType >;
   using ConstCompressedRowLengthsVectorView = typename CompressedRowLengthsVectorView::ConstViewType;
   using ValuesVectorType = Containers::Vector< Real, Device, Index, RealAllocator >;
   using RealAllocatorType = RealAllocator;
   using ViewType = MatrixView< Real, Device, Index >;
   using ConstViewType = MatrixView< std::add_const_t< Real >, Device, Index >;

   Matrix( const RealAllocatorType& allocator = RealAllocatorType() );

   Matrix( const IndexType rows,
           const IndexType columns,
           const RealAllocatorType& allocator = RealAllocatorType() );

   void setDimensions( const IndexType rows,
                       const IndexType columns );

   template< typename Matrix_ >
   void setLike( const Matrix_& matrix );

   IndexType getAllocatedElementsCount() const;

   IndexType getNumberOfNonzeroMatrixElements() const;

   void reset();

   __cuda_callable__
   IndexType getRows() const;

   __cuda_callable__
   IndexType getColumns() const;

   const ValuesVectorType& getValues() const;

   ValuesVectorType& getValues();

   // TODO: parallelize and optimize for sparse matrices
   template< typename Matrix >
   bool operator == ( const Matrix& matrix ) const;

   template< typename Matrix >
   bool operator != ( const Matrix& matrix ) const;

   virtual void save( File& file ) const;

   virtual void load( File& file );

   virtual void print( std::ostream& str ) const;


   // TODO: method for symmetric matrices, should not be in general Matrix interface
   [[deprecated]]
   __cuda_callable__
   const IndexType& getNumberOfColors() const;

   // TODO: method for symmetric matrices, should not be in general Matrix interface
   [[deprecated]]
   void computeColorsVector(Containers::Vector<Index, Device, Index> &colorsVector);

   // TODO: copy should be done in the operator= and it should work the other way too
   void copyFromHostToCuda( Matrices::Matrix< Real, Devices::Host, Index >& matrix );

   // TODO: missing implementation!
   __cuda_callable__
   Index getValuesSize() const;

   protected:

   IndexType rows, columns;

   // TODO: remove1
   IndexType numberOfColors;

   ValuesVectorType values;
};

template< typename Real, typename Device, typename Index >
std::ostream& operator << ( std::ostream& str, const Matrix< Real, Device, Index >& m )
{
   m.print( str );
   return str;
}

template< typename Matrix,
          typename InVector,
          typename OutVector >
void MatrixVectorProductCuda( const Matrix& matrix,
                              const InVector& inVector,
                              OutVector& outVector );

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/Matrix.hpp>
