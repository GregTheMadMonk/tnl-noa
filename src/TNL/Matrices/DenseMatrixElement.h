/***************************************************************************
                          DenseMatrixElement.h -  description
                             -------------------
    begin                : Mar 22, 2021
    copyright            : (C) 2021 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <ostream>

#include <TNL/Cuda/CudaCallable.h>

namespace TNL {
namespace Matrices {


template< typename Real,
          typename Index >
class DenseMatrixElement
{
   public:

      using RealType = Real;

      using IndexType = Index;

      __cuda_callable__
      DenseMatrixElement( RealType& value,
                          const IndexType& rowIdx,
                          const IndexType& columnIdx,
                          const IndexType& localIdx )  // localIdx is here only for compatibility with SparseMatrixElement
      : value_( value ), rowIdx( rowIdx ), columnIdx( columnIdx ) {};

      __cuda_callable__
      RealType& value() { return value_; };

      __cuda_callable__
      const RealType& value() const { return value_; };

      __cuda_callable__
      const IndexType& rowIndex() const { return rowIdx; };

      __cuda_callable__
      const IndexType& columnIndex() const { return columnIdx; };

      __cuda_callable__
      const IndexType& localIndex() const { return columnIdx; };

   protected:

      RealType& value_;

      const IndexType& rowIdx;

      const IndexType columnIdx;
};

   } // namespace Matrices
} // namespace TNL
