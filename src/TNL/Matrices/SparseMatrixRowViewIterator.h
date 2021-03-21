 /***************************************************************************
                          SparseMatrixRowView.h -  description
                             -------------------
    begin                : Dec 28, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <ostream>

#include <TNL/Cuda/CudaCallable.h>
#include <TNL/Matrices/SparseMatrixElement.h>

namespace TNL {
namespace Matrices {

template< typename RowView >
class SparseMatrixRowViewIterator
{

   public:

      /**
       * \brief Type of SparseMatrixRowView
       */
      using RowViewType = RowView;

      /**
       * \brief The type of matrix elements.
       */
      using RealType = typename RowViewType::RealType;

      /**
       * \brief The type used for matrix elements indexing.
       */
      using IndexType = typename RowViewType::IndexType;

      /**
       * \brief The type of related matrix element.
       */
      using MatrixElementType = SparseMatrixElement< RealType, IndexType >;

      /**
       * \brief Tells whether the parent matrix is a binary matrix.
       * @return `true` if the matrix is binary.
       */
      static constexpr bool isBinary() { return RowViewType::isBinary(); };

      __cuda_callable__
      SparseMatrixRowViewIterator( RowViewType& rowView,
                                   const IndexType& localIdx );

      /**
       * \brief Comparison of two matrix row iterators.
       *
       * \param other is another matrix row iterator.
       * \return \e true if both iterators points at the same point of the same matrix, \e false otherwise.
       */
      __cuda_callable__
      bool operator==( const SparseMatrixRowViewIterator& other ) const;

      /**
       * \brief Comparison of two matrix row iterators.
       *
       * \param other is another matrix row iterator.
       * \return \e false if both iterators points at the same point of the same matrix, \e true otherwise.
       */
      __cuda_callable__
      bool operator!=( const SparseMatrixRowViewIterator& other ) const;

      __cuda_callable__
      SparseMatrixRowViewIterator& operator++();

      __cuda_callable__
      SparseMatrixRowViewIterator& operator--();

      __cuda_callable__
      MatrixElementType operator*();

      __cuda_callable__
      const MatrixElementType operator*() const;

   protected:

      RowViewType& rowView;

      IndexType localIdx = 0;
};


   } // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/SparseMatrixRowViewIterator.hpp>
