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


template< typename Real,
          typename Index,
          bool isBinary_ = false >
class SparseMatrixElement
{
   public:

      using RealType = Real;

      using IndexType = Index;

      __cuda_callable__
      SparseMatrixElement( RealType& value,
                           const IndexType& rowIdx,
                           IndexType& columnIdx,
                           const IndexType& localIdx )
      : value_( value ), rowIdx( rowIdx ), columnIdx( columnIdx ), localIdx( localIdx ) {};

      __cuda_callable__
      RealType& value() { return value_; };

      __cuda_callable__
      const RealType& value() const { return value_; };

      __cuda_callable__
      const IndexType& rowIndex() const { return rowIdx; };

      __cuda_callable__
      IndexType& columnIndex() { return columnIdx; };

      __cuda_callable__
      const IndexType& columnIndex() const { return columnIdx; };

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
