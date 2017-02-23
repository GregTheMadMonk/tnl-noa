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

template< typename Real, typename Device, typename Index >
class Dummy
{
   public:

   template< typename Matrix >
   void update( const Matrix& matrix ) {}

   template< typename Vector1, typename Vector2 >
   bool solve( const Vector1& b, Vector2& x ) const
   {
      TNL_ASSERT( false,
              std::cerr << "The solve() method of a dummy preconditioner should not be called." << std::endl; );
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

template< typename LinearSolver, typename Real, typename Device, typename Index >
class SolverStarterSolverPreconditionerSetter< LinearSolver, Dummy< Real, Device, Index > >
{
   public:

      typedef Device DeviceType;
      typedef Dummy< Real, DeviceType, Index > PreconditionerType;
   
      static void run( LinearSolver& solver,
                       SharedPointer< PreconditionerType, typename LinearSolver::DeviceType >& preconditioner )
      {
         // do nothing
      }
};

} // namespace Preconditioners
} // namespace Linear
} // namespace Solvers
} // namespace TNL
