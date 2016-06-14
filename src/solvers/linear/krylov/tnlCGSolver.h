/***************************************************************************
                          tnlCGSolver.h  -  description
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

#ifndef tnlCGSolverH
#define tnlCGSolverH

#include <math.h>
#include <core/tnlObject.h>
#include <core/vectors/tnlVector.h>
#include <core/vectors/tnlSharedVector.h>
#include <solvers/preconditioners/tnlDummyPreconditioner.h>
#include <solvers/tnlIterativeSolver.h>
#include <solvers/linear/tnlLinearResidueGetter.h>

template< typename Matrix,
          typename Preconditioner = tnlDummyPreconditioner< typename Matrix :: RealType,
                                                            typename Matrix :: DeviceType,
                                                            typename Matrix :: IndexType> >
class tnlCGSolver : public tnlObject,
                    public tnlIterativeSolver< typename Matrix :: RealType,
                                               typename Matrix :: IndexType >
{
   public:

   typedef typename Matrix :: RealType RealType;
   typedef typename Matrix :: IndexType IndexType;
   typedef typename Matrix :: DeviceType DeviceType;
   typedef Matrix MatrixType;
   typedef Preconditioner PreconditionerType;
   typedef tnlSharedPointer< MatrixType, DeviceType > MatrixPointer;
   typedef tnlSharedPointer< PreconditionerType, DeviceType > PreconditionerPointer;
   // TODO: make this const


   tnlCGSolver();
   
   tnlString getType() const;

   static void configSetup( tnlConfigDescription& config,
                            const tnlString& prefix = "" );

   bool setup( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" );

   void setMatrix( MatrixPointer& matrix );

   void setPreconditioner( const PreconditionerType& preconditioner );

#ifdef HAVE_NOT_CXX11
   template< typename Vector,
             typename ResidueGetter >
   bool solve( const Vector& b, Vector& x );
#else
   template< typename VectorPointer,
             typename ResidueGetter = tnlLinearResidueGetter< Matrix, VectorPointer >  >
   bool solve( const VectorPointer& b, VectorPointer& x );
#endif

   ~tnlCGSolver();

   protected:

   bool setSize( IndexType size );

   tnlVector< RealType, DeviceType, IndexType >  r, new_r, p, Ap;

   MatrixPointer matrix;
   const PreconditionerType* preconditioner;
};

#include <solvers/linear/krylov/tnlCGSolver_impl.h>

#endif
