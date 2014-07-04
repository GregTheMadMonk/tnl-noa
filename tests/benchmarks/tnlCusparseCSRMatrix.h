/***************************************************************************
                          tnlCusparseCSRMatrix.h  -  description
                             -------------------
    begin                : Jul 3, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include <cusparse.h>
#include <core/tnlAssert.h>
#include <core/tnlCuda.h>

template< typename Real >
class tnlCusparseCSRMatrix
{};

template< typename Real >
class tnlCusparseCSRMatrixBase
{
   public:
      typedef Real RealType;
      typedef tnlCuda DeviceType;
      typedef tnlCSRMatrix< RealType, tnlCuda, int > MatrixType;

      tnlCusparseCSRMatrixBase()
      : matrix( 0 )
      {
      };

      void init( const MatrixType& matrix,
                 cusparseHandle_t* cusparseHandle )
      {
         this->matrix = &matrix;
         this->cusparseHandle = cusparseHandle;
         cusparseCreateMatDescr( & this->matrixDescriptor );
      };

      int getRows() const
      {
         return matrix->getRows();
      }

      int getColumns() const
      {
         return matrix->getColumns();
      }

      int getNumberOfMatrixElements() const
      {
         return matrix->getNumberOfMatrixElements();
      }


      template< typename InVector,
                typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector ) const
      {
         tnlAssert( matrix, );
         cusparseDcsrmv( *( this->cusparseHandle ),
                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                         this->matrix->getRows(),
                         this->matrix->getColumns(),
                         this->matrix->values.getSize(),
                         1.0,
                         this->matrixDescriptor,
                         this->matrix->values.getData(),
                         this->matrix->rowPointers.getData(),
                         this->matrix->columnIndexes.getData(),
                         inVector.getData(),
                         1.0,
                         outVector.getData() );
      }

   protected:

      const MatrixType* matrix;

      cusparseHandle_t* cusparseHandle;

      cusparseMatDescr_t matrixDescriptor;

};


template<>
class tnlCusparseCSRMatrix< double > : public tnlCusparseCSRMatrixBase< double >
{
   public:

      template< typename InVector,
                typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector ) const
      {
         tnlAssert( matrix, );
         cusparseDcsrmv( *( this->cusparseHandle ),
                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                         this->matrix->getRows(),
                         this->matrix->getColumns(),
                         this->matrix->values.getSize(),
                         this->matrixDescriptor,
                         this->matrix->values.getData(),
                         this->matrix->rowPointers.getData(),
                         this->matrix->columnIndexes.getData(),
                         inVector.getData(),
                         1.0,
                         outVector.getData() );
      }
};

template<>
class tnlCusparseCSRMatrix< float > : public tnlCusparseCSRMatrixBase< float >
{
   public:

      template< typename InVector,
                typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector ) const
      {
         tnlAssert( matrix, );
         cusparseScsrmv( *( this->cusparseHandle ),
                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                         this->matrix->getRows(),
                         this->matrix->getColumns(),
                         this->matrix->values.getSize(),                         
                         this->matrixDescriptor,
                         this->matrix->values.getData(),
                         this->matrix->rowPointers.getData(),
                         this->matrix->columnIndexes.getData(),
                         inVector.getData(),
                         0,
                         outVector.getData() );
      }
};

