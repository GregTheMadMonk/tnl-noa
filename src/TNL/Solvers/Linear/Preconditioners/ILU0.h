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

#include "Preconditioner.h"

#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/CSR.h>

#if defined(HAVE_CUDA) && defined(HAVE_CUSPARSE)
#include <cusparse.h>
#endif

namespace TNL {
namespace Solvers {
namespace Linear {
namespace Preconditioners {

// implementation template
template< typename Matrix, typename Real, typename Device, typename Index >
class ILU0_impl
{};

// actual template to be used by users
template< typename Matrix >
class ILU0
: public ILU0_impl< Matrix, typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType >
{
public:
   String getType() const
   {
      return String( "ILU0" );
   }
};

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
   Matrices::CSR< RealType, DeviceType, IndexType > L, U;

   // Specialized methods to distinguish between normal and distributed matrices
   // in the implementation.
   template< typename M >
   static IndexType getMinColumn( const M& m )
   {
      return 0;
   }

   template< typename M >
   static IndexType getMinColumn( const DistributedContainers::DistributedMatrix< M >& m )
   {
      return m.getLocalRowRange().getBegin();
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
   template< typename MatrixT,
             typename = typename std::enable_if< ! std::is_same< DeviceType, typename MatrixT::DeviceType >::value >::type >
   void copyMatrix( const MatrixT& matrix )
   {
      typename MatrixT::CudaType A_tmp;
      A_tmp = matrix;
      Matrices::copySparseMatrix( A, A_tmp );
   }

   template< typename MatrixT,
             typename = typename std::enable_if< std::is_same< DeviceType, typename MatrixT::DeviceType >::value >::type,
             typename = void >
   void copyMatrix( const MatrixT& matrix )
   {
      Matrices::copySparseMatrix( A, matrix );
   }
#endif
};

template< typename Matrix, typename Communicator >
class ILU0_impl< DistributedContainers::DistributedMatrix< Matrix, Communicator >, double, Devices::Cuda, int >
: public Preconditioner< DistributedContainers::DistributedMatrix< Matrix, Communicator > >
{
   using MatrixType = DistributedContainers::DistributedMatrix< Matrix, Communicator >;
public:
   using RealType = double;
   using DeviceType = Devices::Cuda;
   using IndexType = int;
   using typename Preconditioner< MatrixType >::VectorViewType;
   using typename Preconditioner< MatrixType >::ConstVectorViewType;
   using typename Preconditioner< MatrixType >::MatrixPointer;

   virtual void update( const MatrixPointer& matrixPointer ) override
   {
      throw std::runtime_error("ILU0 is not implemented yet for CUDA and distributed matrices.");
   }

   virtual void solve( ConstVectorViewType b, VectorViewType x ) const override
   {
      throw std::runtime_error("ILU0 is not implemented yet for CUDA and distributed matrices.");
   }
};

template< typename Matrix, typename Real, typename Index >
class ILU0_impl< Matrix, Real, Devices::MIC, Index >
: public Preconditioner< Matrix >
{
public:
   using RealType = Real;
   using DeviceType = Devices::MIC;
   using IndexType = Index;
   using typename Preconditioner< Matrix >::VectorViewType;
   using typename Preconditioner< Matrix >::ConstVectorViewType;
   using typename Preconditioner< Matrix >::MatrixPointer;

   virtual void update( const MatrixPointer& matrixPointer ) override
   {
      throw std::runtime_error("Not Iplemented yet for MIC");
   }

   virtual void solve( ConstVectorViewType b, VectorViewType x ) const override
   {
      throw std::runtime_error("Not Iplemented yet for MIC");
   }
};

} // namespace Preconditioners
} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/Preconditioners/ILU0_impl.h>
