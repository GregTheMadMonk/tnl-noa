/***************************************************************************
                          tnlDummyPreconditioner.h  -  description
                             -------------------
    begin                : Oct 19, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Object.h>

namespace TNL {

template< typename Real, typename Device, typename Index >
class tnlDummyPreconditioner
{
   public:

   template< typename Matrix >
   void update( const Matrix& matrix ) {}

   template< typename Vector1, typename Vector2 >
   bool solve( const Vector1& b, Vector2& x ) const { return true; }

   String getType() const
   {
      return String( "tnlDummyPreconditioner" );
   }
};

template< typename LinearSolver, typename Preconditioner >
class tnlSolverStarterSolverPreconditionerSetter
{
    public:
        static void run( LinearSolver& solver, Preconditioner& preconditioner )
        {
            solver.setPreconditioner( preconditioner );
        }
};

template< typename LinearSolver, typename Real, typename Device, typename Index >
class tnlSolverStarterSolverPreconditionerSetter< LinearSolver, tnlDummyPreconditioner< Real, Device, Index > >
{
    public:
        static void run( LinearSolver& solver, tnlDummyPreconditioner< Real, Device, Index >& preconditioner )
        {
            // do nothing
        }
};

} // namespace TNL
