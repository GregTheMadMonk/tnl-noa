/***************************************************************************
                          ILU0.h  -  description
                             -------------------
    begin                : Dec 24, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <type_traits>

#include <TNL/Object.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/CSR.h>

#if defined(HAVE_CUDA) && defined(HAVE_CUSPARSE)
#include <cusparse.h>
#endif

namespace TNL {
namespace Solvers {
namespace Linear {
namespace Preconditioners {

template< typename Real, typename Device, typename Index >
class ILU0
{};

template< typename Real, typename Index >
class ILU0< Real, Devices::Host, Index >
{
public:
   using RealType = Real;
   using DeviceType = Devices::Host;
   using IndexType = Index;

   template< typename MatrixPointer >
   void update( const MatrixPointer& matrixPointer );

   template< typename Vector1, typename Vector2 >
   bool solve( const Vector1& b, Vector2& x ) const;

   String getType() const
   {
      return String( "ILU0" );
   }

protected:
   Matrices::CSR< RealType, DeviceType, IndexType > L;
   Matrices::CSR< RealType, DeviceType, IndexType > U;
};

template<>
class ILU0< double, Devices::Cuda, int >
{
public:
   using RealType = double;
   using DeviceType = Devices::Cuda;
   using IndexType = int;

   ILU0()
   {
#if defined(HAVE_CUDA) && defined(HAVE_CUSPARSE)
      cusparseCreate( &handle );
#endif
   }

   template< typename MatrixPointer >
   void update( const MatrixPointer& matrixPointer );

   template< typename Vector1, typename Vector2 >
   bool solve( const Vector1& b, Vector2& x ) const;

   String getType() const
   {
      return String( "ILU0" );
   }

   ~ILU0()
   {
#if defined(HAVE_CUDA) && defined(HAVE_CUSPARSE)
      resetMatrices();
      cusparseDestroy( handle );
#endif
   }

protected:
#if defined(HAVE_CUDA) && defined(HAVE_CUSPARSE)
   Matrices::CSR< RealType, DeviceType, IndexType > A;
   Containers::Vector< RealType, DeviceType, IndexType > y;

   cusparseHandle_t handle;

   cusparseMatDescr_t descr_A = 0;
   cusparseMatDescr_t descr_L = 0;
   cusparseMatDescr_t descr_U = 0;
   csrilu02Info_t     info_A  = 0;
   csrsv2Info_t       info_L  = 0;
   csrsv2Info_t       info_U  = 0;

   const cusparseSolvePolicy_t policy_A = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
   const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
   const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
   const cusparseOperation_t trans_L  = CUSPARSE_OPERATION_NON_TRANSPOSE;
   const cusparseOperation_t trans_U  = CUSPARSE_OPERATION_NON_TRANSPOSE;

   Containers::Array< char, DeviceType, int > pBuffer;

   // scaling factor for triangular solves
   const double alpha = 1.0;

   void resetMatrices()
   {
      if( descr_A ) {
         cusparseDestroyMatDescr( descr_A );
         descr_A = 0;
      }
      if( descr_L ) {
         cusparseDestroyMatDescr( descr_L );
         descr_L = 0;
      }
      if( descr_U ) {
         cusparseDestroyMatDescr( descr_U );
         descr_U = 0;
      }
      if( info_A ) {
         cusparseDestroyCsrilu02Info( info_A );
         info_A = 0;
      }
      if( info_L ) {
         cusparseDestroyCsrsv2Info( info_L );
         info_L = 0;
      }
      if( info_U ) {
         cusparseDestroyCsrsv2Info( info_U );
         info_U = 0;
      }
      pBuffer.reset();
   }

   // TODO: extend Matrices::copySparseMatrix accordingly
   template< typename Matrix,
             typename = typename std::enable_if< ! std::is_same< DeviceType, typename Matrix::DeviceType >::value >::type >
   void copyMatrix( const Matrix& matrix )
   {
      typename Matrix::CudaType A_tmp;
      A_tmp = matrix;
      Matrices::copySparseMatrix( A, A_tmp );
   }

   template< typename Matrix,
             typename = typename std::enable_if< std::is_same< DeviceType, typename Matrix::DeviceType >::value >::type,
             typename = void >
   void copyMatrix( const Matrix& matrix )
   {
      Matrices::copySparseMatrix( A, matrix );
   }
#endif
};

#ifdef HAVE_MIC
template< typename Real, typename Index >
class ILU0< Real, Devices::MIC, Index >
{
public:
   using RealType = Real;
   using DeviceType = Devices::MIC;
   using IndexType = Index;

   template< typename MatrixPointer >
   void update( const MatrixPointer& matrixPointer )
   {
      throw std::runtime_error("Not Iplemented yet for MIC");
   }

   template< typename Vector1, typename Vector2 >
   bool solve( const Vector1& b, Vector2& x ) const
   {
      throw std::runtime_error("Not Iplemented yet for MIC");
   }

   String getType() const
   {
      return String( "ILU0" );
   }
};

#endif

} // namespace Preconditioners
} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/Preconditioners/ILU0_impl.h>
