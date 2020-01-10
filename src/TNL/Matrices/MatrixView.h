/***************************************************************************
                          MatrixView.h  -  description
                             -------------------
    begin                : Dec 28, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Object.h>
#include <TNL/Allocators/Default.h>
#include <TNL/Devices/Host.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>

namespace TNL {
/**
 * \brief Namespace for matrix formats.
 */
namespace Matrices {

template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int >
class MatrixView : public Object
{
public:
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using CompressedRowLengthsVector = Containers::Vector< IndexType, DeviceType, IndexType >;
   using CompressedRowLengthsVectorView = Containers::VectorView< IndexType, DeviceType, IndexType >;
   using ConstCompressedRowLengthsVectorView = typename CompressedRowLengthsVectorView::ConstViewType;
   using ValuesView = Containers::VectorView< RealType, DeviceType, IndexType >;
   using ViewType = MatrixView< typename std::remove_const< Real >::type, Device, Index >;
   using ConstViewType = MatrixView< typename std::add_const< Real >::type, Device, Index >;

   __cuda_callable__
   MatrixView();

   __cuda_callable__
   MatrixView( const IndexType rows,
               const IndexType columns,
               const ValuesView& values );

   __cuda_callable__
   MatrixView( const MatrixView& view ) = default;

   virtual IndexType getRowLength( const IndexType row ) const = 0;

   // TODO: implementation is not parallel
   // TODO: it would be nice if padding zeros could be stripped
   void getCompressedRowLengths( CompressedRowLengthsVector& rowLengths ) const;

   virtual void getCompressedRowLengths( CompressedRowLengthsVectorView rowLengths ) const;

   IndexType getAllocatedElementsCount() const;

   virtual IndexType getNumberOfNonzeroMatrixElements() const;

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

   virtual Real getElement( const IndexType row,
                            const IndexType column ) const = 0;

   const ValuesView& getValues() const;

   ValuesView& getValues();

   /**
    * \brief Shallow copy of the matrix view.
    *
    * @param view
    * @return 
    */
   __cuda_callable__
   MatrixView& operator=( const MatrixView& view );
   
   // TODO: parallelize and optimize for sparse matrices
   template< typename Matrix >
   bool operator == ( const Matrix& matrix ) const;

   template< typename Matrix >
   bool operator != ( const Matrix& matrix ) const;

   virtual void save( File& file ) const;

   virtual void load( File& file );

   virtual void print( std::ostream& str ) const;


   // TODO: method for symmetric matrices, should not be in general Matrix interface
   __cuda_callable__
   const IndexType& getNumberOfColors() const;

   // TODO: method for symmetric matrices, should not be in general Matrix interface
   void computeColorsVector(Containers::Vector<Index, Device, Index> &colorsVector);

   protected:

   IndexType rows, columns;

   ValuesView values;
};

template< typename Real, typename Device, typename Index >
std::ostream& operator << ( std::ostream& str, const MatrixView< Real, Device, Index >& m )
{
   m.print( str );
   return str;
}

/*
template< typename Matrix,
          typename InVector,
          typename OutVector >
void MatrixVectorProductCuda( const Matrix& matrix,
                              const InVector& inVector,
                              OutVector& outVector );
*/

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/MatrixView.hpp>
