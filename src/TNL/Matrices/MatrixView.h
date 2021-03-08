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
 * \brief Namespace for different matrix formats.
 */
namespace Matrices {

/**
 * \brief Base class for other matrix types views.
 *
 * \tparam Real is a type of matrix elements.
 * \tparam Device is a device where the matrix is allocated.
 * \tparam Index is a type for indexing of the matrix elements.
 */
template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int >
class MatrixView : public Object
{
   public:
      using RowsCapacitiesType = Containers::Vector< Index, Device, Index >;
      using RowsCapacitiesTypeView = Containers::VectorView< Index, Device, Index >;
      using ConstRowsCapacitiesTypeView = typename RowsCapacitiesTypeView::ConstViewType;
      using ValuesView = Containers::VectorView< Real, Device, Index >;

      /**
       * \brief The type of matrix elements.
       */
      using RealType = Real;

      /**
       * \brief The device where the matrix is allocated.
       */
      using DeviceType = Device;

      /**
       * \brief The type used for matrix elements indexing.
       */
      using IndexType = Index;

      /**
       * \brief Type of base matrix view.
       *
       */
      using ViewType = MatrixView< typename std::remove_const< Real >::type, Device, Index >;

      /**
       * \brief Type of base matrix view for constant instances.
       *
       */
      using ConstViewType = MatrixView< typename std::add_const< Real >::type, Device, Index >;

      /**
       * \brief Basic construtor with no parameters.
       */
      __cuda_callable__
      MatrixView();

      /**
       * \brief Constructor with matrix dimensions and matrix elements values.
       *
       * The matrix elements values are passed in a form vector view.
       *
       * @param rows is a number of matrix rows.
       * @param columns is a number of matrix columns.
       * @param values is a vector view with matrix elements values.
       */
      __cuda_callable__
      MatrixView( const IndexType rows,
                  const IndexType columns,
                  const ValuesView& values );

      /**
       * @brief Shallow copy constructor.
       *
       * @param view is an input matrix view.
       */
      __cuda_callable__
      MatrixView( const MatrixView& view ) = default;

      /**
       * \brief Move constructor.
       *
       * @param view is an input matrix view.
       */
      __cuda_callable__
      MatrixView( MatrixView&& view ) = default;

      /**
       * \brief Tells the number of allocated matrix elements.
       *
       * In the case of dense matrices, this is just product of the number of rows and the number of columns.
       * But for other matrix types like sparse matrices, this can be different.
       *
       * \return Number of allocated matrix elements.
       */
      IndexType getAllocatedElementsCount() const;

      /**
       * \brief Computes a current number of nonzero matrix elements.
       *
       * \return number of nonzero matrix elements.
       */
      virtual IndexType getNonzeroElementsCount() const;

      /**
       * \brief Returns number of matrix rows.
       *
       * \return number of matrix row.
       */
      __cuda_callable__
      IndexType getRows() const;

      /**
       * \brief Returns number of matrix columns.
       *
       * @return number of matrix columns.
       */
      __cuda_callable__
      IndexType getColumns() const;

      /**
       * \brief Returns a constant reference to a vector with the matrix elements values.
       *
       * \return constant reference to a vector with the matrix elements values.
       */
      __cuda_callable__
      const ValuesView& getValues() const;

      /**
       * \brief Returns a reference to a vector with the matrix elements values.
       *
       * \return constant reference to a vector with the matrix elements values.
       */
      __cuda_callable__
      ValuesView& getValues();

      /**
       * \brief Shallow copy of the matrix view.
       *
       * \param view is an input matrix view.
       * \return reference to this view.
       */
      __cuda_callable__
      MatrixView& operator=( const MatrixView& view );

      /**
       * \brief Comparison operator with another arbitrary matrix view type.
       *
       * \param matrix is the right-hand side matrix.
       * \return \e true if the RHS matrix is equal, \e false otherwise.
       */
      template< typename Matrix >
      bool operator == ( const Matrix& matrix ) const;

      /**
       * \brief Comparison operator with another arbitrary matrix view type.
       *
       * \param matrix is the right-hand side matrix.
       * \return \e true if the RHS matrix is equal, \e false otherwise.
       */

      template< typename Matrix >
      bool operator != ( const Matrix& matrix ) const;

      /**
       * \brief Method for saving the matrix view to a file.
       *
       * \param file is the output file.
       */
      virtual void save( File& file ) const;

      /**
       * \brief Method for printing the matrix view to output stream.
       *
       * \param str is the output stream.
       */
      virtual void print( std::ostream& str ) const;


      // TODO: method for symmetric matrices, should not be in general Matrix interface
      //[[deprecated]]
      //__cuda_callable__
      //const IndexType& getNumberOfColors() const;

      // TODO: method for symmetric matrices, should not be in general Matrix interface
      //[[deprecated]]
      //void computeColorsVector(Containers::Vector<Index, Device, Index> &colorsVector);

      protected:

      IndexType rows, columns;

      ValuesView values;
};

/**
 * \brief Overloaded insertion operator for printing a matrix to output stream.
 *
 * \tparam Real is a type of the matrix elements.
 * \tparam Device is a device where the matrix is allocated.
 * \tparam Index is a type used for the indexing of the matrix elements.
 *
 * \param str is a output stream.
 * \param matrix is the matrix to be printed.
 *
 * \return a reference on the output stream \ref std::ostream&.
 */
template< typename Real, typename Device, typename Index >
std::ostream& operator << ( std::ostream& str, const MatrixView< Real, Device, Index >& m )
{
   m.print( str );
   return str;
}

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/MatrixView.hpp>
