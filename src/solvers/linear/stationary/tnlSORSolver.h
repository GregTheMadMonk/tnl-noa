/***************************************************************************
                          tnlSORSolver.h  -  description
                             -------------------
    begin                : 2007/07/30
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

#ifndef tnlSORSolverH
#define tnlSORSolverH

#include <math.h>
#include <core/tnlObject.h>
#include <solvers/preconditioners/tnlDummyPreconditioner.h>
#include <solvers/tnlIterativeSolver.h>
#include <solvers/linear/tnlResidueGetter.h>


template< typename Matrix,
          typename Preconditioner = tnlDummyPreconditioner< typename Matrix :: RealType,
                                                            typename Matrix :: Device,
                                                            typename Matrix :: IndexType> >
class tnlSORSolver : public tnlObject,
                     public tnlIterativeSolver< typename Matrix :: RealType,
                                                typename Matrix :: IndexType >
{

   typedef typename Matrix :: RealType RealType;
   typedef typename Matrix :: IndexType IndexType;
   typedef typename Matrix :: Device Device;
   typedef Matrix MatrixType;
   typedef Preconditioner PreconditionerType;

   public:

   tnlSORSolver();

   tnlString getType() const;


   void setOmega( const RealType& omega );

   conat RealType& getOmega() const;

   void setMatrix( const MatrixType& matrix );

   void setPreconditioner( const Preconditioner& preconditioner );

   template< typename Vector,
             typename ResidueGetter = tnlLinearResidueGetter< Matrix, Vector > >
   bool solve( const Vector& b, Vector& x );

   ~tnlSORSolver();

   protected:

   RealType omega;

   const MatrixType* matrix;

   const PreconditionerType* preconditioner;

}

#include <solvers/linear/stationary/implementation/tnlSORSolver_impl.h>

#endif
