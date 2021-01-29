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
   using RowsCapacitiesType = Containers::Vector< IndexType, DeviceType, IndexType >;
   using RowsCapacitiesTypeView = Containers::VectorView< IndexType, DeviceType, IndexType >;
   using ConstRowsCapacitiesTypeView = typename RowsCapacitiesTypeView::ConstViewType;
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

   __cuda_callable__
   MatrixView( MatrixView&& view ) = default;

   IndexType getAllocatedElementsCount() const;

   virtual IndexType getNonzeroElementsCount() const;

   __cuda_callable__
   IndexType getRows() const;

   __cuda_callable__
   IndexType getColumns() const;

   __cuda_callable__
   const ValuesView& getValues() const;

   __cuda_callable__
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

   virtual void print( std::ostream& str ) const;


   // TODO: method for symmetric matrices, should not be in general Matrix interface
   [[deprecated]]
   __cuda_callable__
   const IndexType& getNumberOfColors() const;

   // TODO: method for symmetric matrices, should not be in general Matrix interface
   [[deprecated]]
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

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/MatrixView.hpp>
