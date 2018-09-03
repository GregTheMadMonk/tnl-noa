/***************************************************************************
                          Dummy.h  -  description
                             -------------------
    begin                : Oct 19, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Object.h>
#include <TNL/SharedPointer.h>

namespace TNL {
namespace Solvers {
namespace Linear {
namespace Preconditioners {

template< typename Matrix >
class Dummy
{
public:
   void update( const Matrix& matrix ) {}

   template< typename Vector1, typename Vector2 >
   bool solve( const Vector1& b, Vector2& x ) const
   {
      TNL_ASSERT_TRUE( false, "The solve() method of a dummy preconditioner should not be called." );
      return true;
   }

   String getType() const
   {
      return String( "Dummy" );
   }
};

template< typename LinearSolver, typename Preconditioner >
class SolverStarterSolverPreconditionerSetter
{
public:
   static void run( LinearSolver& solver,
                    SharedPointer< Preconditioner, typename LinearSolver::DeviceType >& preconditioner )
   {
      solver.setPreconditioner( preconditioner );
   }
};

template< typename LinearSolver >
class SolverStarterSolverPreconditionerSetter< LinearSolver, Dummy< typename LinearSolver::MatrixType > >
{
public:
   static void run( LinearSolver& solver,
                    SharedPointer< Dummy< typename LinearSolver::MatrixType >, typename LinearSolver::DeviceType >& preconditioner )
   {
      // do nothing
   }
};

} // namespace Preconditioners
} // namespace Linear
} // namespace Solvers
} // namespace TNL
