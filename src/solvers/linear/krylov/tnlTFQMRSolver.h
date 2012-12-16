/***************************************************************************
                          tnlTFQMRSolver.h  -  description
                             -------------------
    begin                : Dec 8, 2012
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

#ifndef tnlTFQMRSolverH
#define tnlTFQMRSolverH

#include <math.h>
#include <core/tnlObject.h>
#include <core/tnlVector.h>
#include <core/tnlSharedVector.h>
#include <solvers/preconditioners/tnlDummyPreconditioner.h>
#include <solvers/tnlIterativeSolver.h>
#include <solvers/linear/tnlLinearResidueGetter.h>

template< typename Matrix,
          typename Preconditioner = tnlDummyPreconditioner< typename Matrix :: RealType,
                                                            typename Matrix :: Device,
                                                            typename Matrix :: IndexType> >

class tnlTFQMRSolver : public tnlObject,
                       public tnlIterativeSolver< typename Matrix :: RealType,
                                                  typename Matrix :: IndexType >
{
   public:

   typedef typename Matrix :: RealType RealType;
   typedef typename Matrix :: IndexType IndexType;
   typedef typename Matrix :: Device Device;
   typedef Matrix MatrixType;
   typedef Preconditioner PreconditionerType;

   public:

   tnlTFQMRSolver();

   tnlString getType() const;

   void setMatrix( const MatrixType& matrix );

   void setPreconditioner( const Preconditioner& preconditioner );

   template< typename Vector,
             typename ResidueGetter = tnlLinearResidueGetter< Matrix, Vector >  >
   bool solve( const Vector& b, Vector& x );

   ~tnlTFQMRSolver();

   protected:

   bool setSize( IndexType size );

   tnlVector< RealType, Device, IndexType >  d, r, w, u, v, r_ast, u_new, Au, Au_new;

   const MatrixType* matrix;
   const PreconditionerType* preconditioner;
};

#include <solvers/linear/krylov/implementation/tnlTFQMRSolver_impl.h>

#endif
