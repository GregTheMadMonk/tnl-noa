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
      : rows( 0 ), columns( 0 ), nonEmptyRows( 0 ){};

      __cuda_callable__
      TridiagonalMatrixIndexer( const IndexType& rows, const IndexType& columns )
      : rows( rows ), columns( columns ), nonEmptyRows( TNL::min( rows, columns ) + ( rows > columns ) ) {};

      __cuda_callable__
      TridiagonalMatrixIndexer( const TridiagonalMatrixIndexer& indexer )
      : rows( indexer.rows ), columns( indexer.columns ), nonEmptyRows( indexer.nonEmptyRows ) {};

      void setDimensions( const IndexType& rows, const IndexType& columns )
      {
         this->rows = rows;
         this->columns = columns;
         this->nonEmptyRows = min( rows, columns ) + ( rows > columns );
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
      const IndexType& getRows() const { return this->rows; };

      __cuda_callable__
      const IndexType& getColumns() const { return this->columns; };

      __cuda_callable__
      const IndexType& getNonEmptyRowsCount() const { return this->nonEmptyRows; };
      __cuda_callable__
      IndexType getStorageSize() const { return 3 * this->nonEmptyRows; };

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
            return localIdx * nonEmptyRows + rowIdx;
      };

      protected:

         IndexType rows, columns, nonEmptyRows;
};
      } //namespace details
   } // namespace Materices
} // namespace TNL
