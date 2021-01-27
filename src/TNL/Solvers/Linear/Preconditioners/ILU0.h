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

#include "Preconditioner.h"

#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Pointers/UniquePointer.h>
#include <TNL/Exceptions/NotImplementedError.h>

#if defined(HAVE_CUDA) && defined(HAVE_CUSPARSE)
#include <TNL/Matrices/Legacy/CSR.h>
#include <cusparse.h>
#endif

namespace TNL {
namespace Solvers {
namespace Linear {
namespace Preconditioners {

// implementation template
template< typename Matrix, typename Real, typename Device, typename Index >
class ILU0_impl
: public Preconditioner< Matrix >
{
public:
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using typename Preconditioner< Matrix >::VectorViewType;
   using typename Preconditioner< Matrix >::ConstVectorViewType;
   using typename Preconditioner< Matrix >::MatrixPointer;

   virtual void update( const MatrixPointer& matrixPointer ) override
   {
      throw Exceptions::NotImplementedError("ILU0 is not implemented yet for the matrix type " + getType< Matrix >());
   }

   virtual void solve( ConstVectorViewType b, VectorViewType x ) const override
   {
      throw Exceptions::NotImplementedError("ILU0 is not implemented yet for the matrix type " + getType< Matrix >());
   }
};

// actual template to be used by users
template< typename Matrix >
class ILU0
: public ILU0_impl< Matrix, typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType >
{};

template< typename Matrix, typename Real, typename Index >
class ILU0_impl< Matrix, Real, Devices::Host, Index >
: public Preconditioner< Matrix >
{
public:
   using RealType = Real;
   using DeviceType = Devices::Host;
   using IndexType = Index;
   using typename Preconditioner< Matrix >::VectorViewType;
   using typename Preconditioner< Matrix >::ConstVectorViewType;
   using typename Preconditioner< Matrix >::MatrixPointer;

   virtual void update( const MatrixPointer& matrixPointer ) override;

   virtual void solve( ConstVectorViewType b, VectorViewType x ) const override;

protected:
   // The factors L and U are stored separately and the rows of U are reversed.
   Matrices::SparseMatrix< RealType, DeviceType, IndexType, Matrices::GeneralMatrix, Algorithms::Segments::CSRDefault > L, U;

   // Specialized methods to distinguish between normal and distributed matrices
   // in the implementation.
   template< typename M >
   static IndexType getMinColumn( const M& m )
   {
      return 0;
   }

   template< typename M >
   static IndexType getMinColumn( const Matrices::DistributedMatrix< M >& m )
   {
      if( m.getRows() == m.getColumns() )
         // square matrix, assume global column indices
         return m.getLocalRowRange().getBegin();
      else
         // non-square matrix, assume ghost indexing
         return 0;
   }
};

template< typename Matrix >
class ILU0_impl< Matrix, double, Devices::Cuda, int >
: public Preconditioner< Matrix >
{
public:
   using RealType = double;
   using DeviceType = Devices::Cuda;
   using IndexType = int;
   using typename Preconditioner< Matrix >::VectorViewType;
   using typename Preconditioner< Matrix >::ConstVectorViewType;
   using typename Preconditioner< Matrix >::MatrixPointer;

   ILU0_impl()
   {
#if defined(HAVE_CUDA) && defined(HAVE_CUSPARSE)
      cusparseCreate( &handle );
#endif
   }

   virtual void update( const MatrixPointer& matrixPointer ) override;

   virtual void solve( ConstVectorViewType b, VectorViewType x ) const override;

   ~ILU0_impl()
   {
#if defined(HAVE_CUDA) && defined(HAVE_CUSPARSE)
      resetMatrices();
      cusparseDestroy( handle );
#endif
   }

   // must be public because nvcc does not allow extended lambdas in private or protected regions
   void allocate_LU();
   void copy_triangular_factors();
protected:

#if defined(HAVE_CUDA) && defined(HAVE_CUSPARSE)
   using CSR = Matrices::Legacy::CSR< RealType, DeviceType, IndexType >;
   Pointers::UniquePointer< CSR > A, L, U;
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
//   const cusparseSolvePolicy_t policy_A = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
//   const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
//   const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
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
#endif
};

template< typename Matrix >
class ILU0_impl< Matrices::DistributedMatrix< Matrix >, double, Devices::Cuda, int >
: public Preconditioner< Matrices::DistributedMatrix< Matrix > >
{
   using MatrixType = Matrices::DistributedMatrix< Matrix >;
public:
   using RealType = double;
   using DeviceType = Devices::Cuda;
   using IndexType = int;
   using typename Preconditioner< MatrixType >::VectorViewType;
   using typename Preconditioner< MatrixType >::ConstVectorViewType;
   using typename Preconditioner< MatrixType >::MatrixPointer;

   virtual void update( const MatrixPointer& matrixPointer ) override
   {
      throw Exceptions::NotImplementedError("ILU0 is not implemented yet for CUDA and distributed matrices.");
   }

   virtual void solve( ConstVectorViewType b, VectorViewType x ) const override
   {
      throw Exceptions::NotImplementedError("ILU0 is not implemented yet for CUDA and distributed matrices.");
   }
};

} // namespace Preconditioners
} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/Preconditioners/ILU0_impl.h>
