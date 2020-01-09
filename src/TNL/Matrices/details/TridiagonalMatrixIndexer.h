/***************************************************************************
                          TridiagonalMatrixIndexer.h  -  description
                             -------------------
    begin                : Jan 9, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
   namespace Matrices {
      namespace details {

template< typename Index,
          bool RowMajorOrder >
class TridiagonalMatrixIndexer
{
   public:

      using IndexType = Index;

      static constexpr bool getRowMajorOrder() { return RowMajorOrder; };

      __cuda_callable__
      TridiagonalMatrixIndexer()
      : rows( 0 ), columns( 0 ), size( 0 ){};

      __cuda_callable__
      TridiagonalMatrixIndexer( const IndexType& rows, const IndexType& columns )
      : rows( rows ), columns( columns ), size( TNL::min( rows, columns ) ) {};

      __cuda_callable__
      TridiagonalMatrixIndexer( const TridiagonalMatrixIndexer& indexer )
      : rows( indexer.rows ), columns( indexer.columns ), size( indexer.size ) {};

      void setDimensions( const IndexType& rows, const IndexType& columns )
      {
         this->rows = rows;
         this->columns = columns;
         this->size = min( rows, columns );
      };

      __cuda_callable__
      IndexType getRowSize( const IndexType rowIdx ) const
      {
         if( rowIdx == 0 )
            return 2;
         if( columns <= rows )
         {
            if( rowIdx == columns - 1 )
               return 2;
            if( rowIdx == columns )
               return 1;
         }
         return 3;
      };

      __cuda_callable__
      IndexType getRows() const { return this->rows; };

      __cuda_callable__
      IndexType getColumns() const { return this->rows; };

      __cuda_callable__
      IndexType getStorageSize() const { return 3 * this->size; };

      __cuda_callable__
      IndexType getGlobalIndex( const Index rowIdx, const Index localIdx ) const
      {
         TNL_ASSERT_GE( localIdx, 0, "" );
         TNL_ASSERT_LT( localIdx, 3, "" );
         TNL_ASSERT_GE( rowIdx, 0, "" );
         TNL_ASSERT_LT( rowIdx, this->rows, "" );
         
         if( RowMajorOrder )
            return 3 * rowIdx + localIdx;
         else
            return localIdx * size + rowIdx;
      };

      protected:

         IndexType rows, columns, size;
};
      } //namespace details
   } // namespace Materices
} // namespace TNL
