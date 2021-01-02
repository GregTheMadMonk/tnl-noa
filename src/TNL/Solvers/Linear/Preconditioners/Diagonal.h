/***************************************************************************
                          Diagonal.h  -  description
                             -------------------
    begin                : Dec 17, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include "Preconditioner.h"

#include <TNL/Containers/Vector.h>

namespace TNL {
namespace Solvers {
namespace Linear {
namespace Preconditioners {

template< typename Matrix >
class Diagonal
: public Preconditioner< Matrix >
{
public:
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using typename Preconditioner< Matrix >::VectorViewType;
   using typename Preconditioner< Matrix >::ConstVectorViewType;
   using typename Preconditioner< Matrix >::MatrixPointer;
   using VectorType = Containers::Vector< RealType, DeviceType, IndexType >;

   virtual void update( const MatrixPointer& matrixPointer ) override;

   virtual void solve( ConstVectorViewType b, VectorViewType x ) const override;

protected:
   VectorType diagonal;
};

template< typename Matrix >
class Diagonal< Matrices::DistributedMatrix< Matrix > >
: public Preconditioner< Matrices::DistributedMatrix< Matrix > >
{
public:
   using MatrixType = Matrices::DistributedMatrix< Matrix >;
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;
   using typename Preconditioner< MatrixType >::VectorViewType;
   using typename Preconditioner< MatrixType >::ConstVectorViewType;
   using typename Preconditioner< MatrixType >::MatrixPointer;
   using VectorType = Containers::Vector< RealType, DeviceType, IndexType >;
   using LocalViewType = Containers::VectorView< RealType, DeviceType, IndexType >;
   using ConstLocalViewType = Containers::VectorView< std::add_const_t< RealType >, DeviceType, IndexType >;

   virtual void update( const MatrixPointer& matrixPointer ) override;

   virtual void solve( ConstVectorViewType b, VectorViewType x ) const override;

protected:
   VectorType diagonal;
};

} // namespace Preconditioners
} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/Preconditioners/Diagonal_impl.h>
