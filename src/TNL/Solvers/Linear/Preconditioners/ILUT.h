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

#include <type_traits>

#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/CSR.h>

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
   String getType() const
   {
      return String( "ILUT" );
   }
};

template< typename Matrix, typename Real, typename Index >
class ILUT_impl< Matrix, Real, Devices::Host, Index >
{
public:
   using RealType = Real;
   using DeviceType = Devices::Host;
   using IndexType = Index;
   using VectorType = Containers::Vector< RealType, DeviceType, IndexType >;

// TODO: setup parameters from CLI
//   ILUT( Index p, Real tau ) : p(p), tau(tau) {}

   void update( const Matrix& matrix );

   template< typename Vector1, typename Vector2 >
   bool solve( const Vector1& b, Vector2& x ) const;

protected:
   Index p = 8;
   Real tau = 1e-4;

   // The factors L and U are stored separately and the rows of U are reversed.
   Matrices::CSR< RealType, DeviceType, IndexType > L;
   Matrices::CSR< RealType, DeviceType, IndexType > U;
};

template< typename Matrix, typename Real, typename Index >
class ILUT_impl< Matrix, Real, Devices::Cuda, Index >
{
public:
   using RealType = Real;
   using DeviceType = Devices::Cuda;
   using IndexType = Index;

   void update( const Matrix& matrix )
   {
      throw std::runtime_error("Not Iplemented yet for CUDA");
   }

   template< typename Vector1, typename Vector2 >
   bool solve( const Vector1& b, Vector2& x ) const
   {
      throw std::runtime_error("Not Iplemented yet for CUDA");
   }
};

template< typename Matrix, typename Real, typename Index >
class ILUT_impl< Matrix, Real, Devices::MIC, Index >
{
public:
   using RealType = Real;
   using DeviceType = Devices::MIC;
   using IndexType = Index;

   void update( const Matrix& matrix )
   {
      throw std::runtime_error("Not Iplemented yet for MIC");
   }

   template< typename Vector1, typename Vector2 >
   bool solve( const Vector1& b, Vector2& x ) const
   {
      throw std::runtime_error("Not Iplemented yet for MIC");
   }
};

} // namespace Preconditioners
} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/Preconditioners/ILUT_impl.h>
