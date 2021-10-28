/***************************************************************************
                          SparseMatrixElement.h -  description
                             -------------------
    begin                : Mar 21, 2021
    copyright            : (C) 2021 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <ostream>

#include <TNL/Cuda/CudaCallable.h>

namespace TNL {
namespace Matrices {


/**
 * \brief Accessor for sparse matrix elements.
 *
 * \tparam Real is a type of matrix elements values.
 * \tparam Index is a type of matrix elements column indexes.
 */
template< typename Real,
          typename Index >
class SparseMatrixElement
{
   public:

      /**
       * \brief Test of binary matrix type.
       *
       * \return \e true if the matrix is stored as binary and \e false otherwise.
       */
      static constexpr bool isBinary() { return std::is_same< std::remove_const_t< Real >, bool >::value; };

      /**
       * \brief Type of matrix elements values.
       */
      using RealType = Real;

      /**
       * \brief Type of matrix elements column indexes.
       */
      using IndexType = Index;

      /**
       * \brief Constructor.
       *
       * \param value is matrix element value.
       * \param rowIdx is row index of the matrix element.
       * \param columnIdx is a column index of the matrix element.
       * \param localIdx is the rank of the non-zero elements in the matrix row.
       */
      __cuda_callable__
      SparseMatrixElement( RealType& value,
                           const IndexType& rowIdx,
                           IndexType& columnIdx,
                           const IndexType& localIdx )
      : value_( value ), rowIdx( rowIdx ), columnIdx( columnIdx ), localIdx( localIdx ) {};

      /**
       * \brief Returns reference on matrix element value.
       *
       * \return reference on matrix element value.
       */
      __cuda_callable__
      RealType& value() { return value_; };

      /**
       * \brief Returns constant reference on matrix element value.
       *
       * \return constant reference on matrix element value.
       */
      __cuda_callable__
      const RealType& value() const { return value_; };

      /**
       * \brief Returns constant reference on matrix element column index.
       *
       * \return constant reference on matrix element column index.
       */
      __cuda_callable__
      const IndexType& rowIndex() const { return rowIdx; };

      /**
       * \brief Returns reference on matrix element column index.
       *
       * \return reference on matrix element column index.
       */
      __cuda_callable__
      IndexType& columnIndex() { return columnIdx; };

      /**
       * \brief Returns constant reference on matrix element column index.
       *
       * \return constant reference on matrix element column index.
       */
      __cuda_callable__
      const IndexType& columnIndex() const { return columnIdx; };

      /**
       * \brief Returns constant reference on the rank of the non-zero matrix element in the row.
       *
       * \return constant reference on the rank of the non-zero matrix element in the row.
       */
      __cuda_callable__
      const IndexType& localIndex() const { return localIdx; };

   protected:

      RealType& value_;

      const IndexType& rowIdx;

      IndexType& columnIdx;

      const IndexType& localIdx;
};

   } // namespace Matrices
} // namespace TNL
