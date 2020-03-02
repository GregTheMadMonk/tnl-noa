/***************************************************************************
                          Sparse.h  -  description
                             -------------------
    begin                : Dec 21, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Matrices/Matrix.h>
#include <TNL/Matrices/Legacy/SparseRow.h>

namespace TNL {
namespace Matrices {

template< typename Real,
          typename Device,
          typename Index >
class Sparse : public Matrix< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename Matrix< RealType, DeviceType, IndexType >::ValuesHolderType ValuesVector;
   typedef Containers::Vector< IndexType, DeviceType, IndexType > ColumnIndexesVector;
   typedef Matrix< Real, Device, Index > BaseType;
   typedef SparseRow< RealType, IndexType > MatrixRow;
   typedef SparseRow< const RealType, const IndexType > ConstMatrixRow;

   Sparse();

   template< typename Real2, typename Device2, typename Index2 >
   void setLike( const Sparse< Real2, Device2, Index2 >& matrix );

   IndexType getNumberOfNonzeroMatrixElements() const;

   IndexType getMaxRowLength() const;

   __cuda_callable__
   IndexType getPaddingIndex() const;

   void reset();

   void save( File& file ) const;

   void load( File& file );

   void printStructure( std::ostream& str ) const;

   protected:

   void allocateMatrixElements( const IndexType& numberOfMatrixElements );

   Containers::Vector< Index, Device, Index > columnIndexes;

   Index maxRowLength;
};

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/Legacy/Sparse_impl.h>
#include <TNL/Matrices/SparseOperations.h>
