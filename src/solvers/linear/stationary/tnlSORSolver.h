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
#include <core/tnlSharedPointer.h>
#include <solvers/preconditioners/tnlDummyPreconditioner.h>
#include <solvers/tnlIterativeSolver.h>
#include <solvers/linear/tnlLinearResidueGetter.h>


template< typename Matrix,
          typename Preconditioner = tnlDummyPreconditioner< typename Matrix :: RealType,
                                                            typename Matrix :: DeviceType,
                                                            typename Matrix :: IndexType> >
class tnlSORSolver : public tnlObject,
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
   // TODO: make this: typedef tnlSharedPointer< const MatrixType, DeviceType > ConstMatrixPointer; 


   tnlSORSolver();

   tnlString getType() const;

   static void configSetup( tnlConfigDescription& config,
                            const tnlString& prefix = "" );

   bool setup( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" );

   void setOmega( const RealType& omega );

   const RealType& getOmega() const;

   void setMatrix( MatrixPointer& matrix );

   void setPreconditioner( const Preconditioner& preconditioner );

#ifdef HAVE_NOT_CXX11
   template< typename VectorPointer,
             typename ResidueGetter >
   bool solve( const VectorPointer& b, VectorPointer& x );
#else
   template< typename VectorPointer,
             typename ResidueGetter = tnlLinearResidueGetter< MatrixPointer, VectorPointer > >
   bool solve( const VectorPointer& b, VectorPointer& x );
#endif   

   ~tnlSORSolver();

   protected:

   RealType omega;

   MatrixPointer matrix;

   const PreconditionerType* preconditioner;

};

#include <solvers/linear/stationary/tnlSORSolver_impl.h>

#endif
