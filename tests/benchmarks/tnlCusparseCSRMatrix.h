/***************************************************************************
                          tnlCusparseCSR.h  -  description
                             -------------------
    begin                : Jul 3, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Assert.h>
#include <TNL/Devices/Cuda.h>
#ifdef HAVE_CUDA
#include <cusparse.h>
#endif

namespace TNL {

template< typename Real >
class tnlCusparseCSRBase
{
   public:
      typedef Real RealType;
      typedef Devices::Cuda DeviceType;
      typedef Matrices::CSR< RealType, Devices::Cuda, int > MatrixType;

      tnlCusparseCSRBase()
      : matrix( 0 )
      {
      };

#ifdef HAVE_CUDA
      void init( const MatrixType& matrix,
                 cusparseHandle_t* cusparseHandle )
      {
         this->matrix = &matrix;
         this->cusparseHandle = cusparseHandle;
         cusparseCreateMatDescr( & this->matrixDescriptor );
      };
#endif

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
         TNL_ASSERT( matrix, );
#ifdef HAVE_CUDA
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
#endif
      }

   protected:

      const MatrixType* matrix;
#ifdef HAVE_CUDA
      cusparseHandle_t* cusparseHandle;

      cusparseMatDescr_t matrixDescriptor;
#endif
};


template< typename Real >
class tnlCusparseCSR
{};

template<>
class tnlCusparseCSR< double > : public tnlCusparseCSRBase< double >
{
   public:

      template< typename InVector,
                typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector ) const
      {
         tnlAssert( matrix, );
#ifdef HAVE_CUDA  
	 double d = 1.0;       
         double* alpha = &d;
         cusparseDcsrmv( *( this->cusparseHandle ),
                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                         this->matrix->getRows(),
                         this->matrix->getColumns(),
                         this->matrix->values.getSize(),
                         alpha,
                         this->matrixDescriptor,
                         this->matrix->values.getData(),
                         this->matrix->rowPointers.getData(),
                         this->matrix->columnIndexes.getData(),
                         inVector.getData(),
                         alpha,
                         outVector.getData() );
#endif         
      }
};

template<>
class tnlCusparseCSR< float > : public tnlCusparseCSRBase< float >
{
   public:

      template< typename InVector,
                typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector ) const
      {
         tnlAssert( matrix, );
#ifdef HAVE_CUDA         
         float d = 1.0;       
         float* alpha = &d;
         cusparseScsrmv( *( this->cusparseHandle ),
                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                         this->matrix->getRows(),
                         this->matrix->getColumns(),
                         this->matrix->values.getSize(),
                         alpha,
                         this->matrixDescriptor,
                         this->matrix->values.getData(),
                         this->matrix->rowPointers.getData(),
                         this->matrix->columnIndexes.getData(),
                         inVector.getData(),
                         alpha,
                         outVector.getData() );
#endif         
      }
};

} // namespace TNL

