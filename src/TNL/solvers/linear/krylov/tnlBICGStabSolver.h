/***************************************************************************
                          tnlBICGStabSolver.h  -  description
                             -------------------
    begin                : 2007/07/31
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <math.h>
#include <TNL/tnlObject.h>
#include <TNL/core/vectors/tnlVector.h>
#include <TNL/core/vectors/tnlSharedVector.h>
#include <TNL/solvers/preconditioners/tnlDummyPreconditioner.h>
#include <TNL/solvers/tnlIterativeSolver.h>
#include <TNL/solvers/linear/tnlLinearResidueGetter.h>

namespace TNL {

template< typename Matrix,
          typename Preconditioner = tnlDummyPreconditioner< typename Matrix :: RealType,
                                                            typename Matrix :: DeviceType,
                                                            typename Matrix :: IndexType> >

class tnlBICGStabSolver : public tnlObject,
                          public tnlIterativeSolver< typename Matrix :: RealType,
                                                     typename Matrix :: IndexType >
{
   public:

   typedef typename Matrix :: RealType RealType;
   typedef typename Matrix :: IndexType IndexType;
   typedef typename Matrix :: DeviceType Device;
   typedef Matrix MatrixType;
   typedef Preconditioner PreconditionerType;

   public:

   tnlBICGStabSolver();

   tnlString getType() const;

   static void configSetup( tnlConfigDescription& config,
                            const tnlString& prefix = "" );

   bool setup( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" );

   void setMatrix( const MatrixType& matrix );

   void setPreconditioner( const Preconditioner& preconditioner );

#ifdef HAVE_NOT_CXX11
   template< typename Vector,
             typename ResidueGetter  >
   bool solve( const Vector& b, Vector& x );
#else
   template< typename Vector,
             typename ResidueGetter = tnlLinearResidueGetter< Matrix, Vector >  >
   bool solve( const Vector& b, Vector& x );
#endif

   ~tnlBICGStabSolver();

   protected:

   bool setSize( IndexType size );

   tnlVector< RealType, Device, IndexType >  r, r_ast, r_new, p, s, Ap, As, M_tmp;

   const MatrixType* matrix;
   const PreconditionerType* preconditioner;
};

} // namespace TNL

#include <TNL/solvers/linear/krylov/tnlBICGStabSolver_impl.h>
