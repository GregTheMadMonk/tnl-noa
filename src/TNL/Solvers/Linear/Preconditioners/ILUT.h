/***************************************************************************
                          ILUT.h  -  description
                             -------------------
    begin                : Aug 31, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include "Preconditioner.h"

#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/SparseMatrix.h>

namespace TNL {
namespace Solvers {
namespace Linear {
namespace Preconditioners {

// implementation template
template< typename Matrix, typename Real, typename Device, typename Index >
class ILUT_impl
{};

// actual template to be used by users
template< typename Matrix >
class ILUT
: public ILUT_impl< Matrix, typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType >
{
public:
   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" )
   {
      config.addEntry< int >( prefix + "ilut-p", "Number of additional non-zero entries to allocate on each row of the factors L and U.", 0 );
      config.addEntry< double >( prefix + "ilut-threshold", "Threshold for droppping small entries.", 1e-4 );
   }
};

template< typename Matrix, typename Real, typename Index >
class ILUT_impl< Matrix, Real, Devices::Host, Index >
: public Preconditioner< Matrix >
{
public:
   using RealType = Real;
   using DeviceType = Devices::Host;
   using IndexType = Index;
   using typename Preconditioner< Matrix >::VectorViewType;
   using typename Preconditioner< Matrix >::ConstVectorViewType;
   using typename Preconditioner< Matrix >::MatrixPointer;
   using VectorType = Containers::Vector< RealType, DeviceType, IndexType >;

   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" ) override;

   virtual void update( const MatrixPointer& matrixPointer ) override;

   virtual void solve( ConstVectorViewType b, VectorViewType x ) const override;

protected:
   Index p = 0;
   Real tau = 1e-4;

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

template< typename Matrix, typename Real, typename Index >
class ILUT_impl< Matrix, Real, Devices::Cuda, Index >
: public Preconditioner< Matrix >
{
public:
   using RealType = Real;
   using DeviceType = Devices::Cuda;
   using IndexType = Index;
   using typename Preconditioner< Matrix >::VectorViewType;
   using typename Preconditioner< Matrix >::ConstVectorViewType;
   using typename Preconditioner< Matrix >::MatrixPointer;

   virtual void update( const MatrixPointer& matrixPointer ) override
   {
      throw std::runtime_error("Not Iplemented yet for CUDA");
   }

   virtual void solve( ConstVectorViewType b, VectorViewType x ) const override
   {
      throw std::runtime_error("Not Iplemented yet for CUDA");
   }
};

} // namespace Preconditioners
} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include "ILUT.hpp"
