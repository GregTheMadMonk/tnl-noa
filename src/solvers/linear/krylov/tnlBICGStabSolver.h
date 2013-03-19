/***************************************************************************
                          tnlBICGStabSolver.h  -  description
                             -------------------
    begin                : 2007/07/31
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef tnlBICGStabSolverH
#define tnlBICGStabSolverH

#include <math.h>
#include <core/tnlObject.h>
#include <core/tnlVector.h>
#include <core/tnlSharedVector.h>
#include <solvers/preconditioners/tnlDummyPreconditioner.h>
#include <solvers/tnlIterativeSolver.h>
#include <solvers/linear/tnlLinearResidueGetter.h>

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

#include <implementation/solvers/linear/krylov/tnlBICGStabSolver_impl.h>

#endif
