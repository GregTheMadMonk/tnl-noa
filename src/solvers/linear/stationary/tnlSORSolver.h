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
#include <solvers/linear/tnlLinearResidueGetter.h>


template< typename Matrix,
          typename Preconditioner = tnlDummyPreconditioner< typename Matrix :: RealType,
                                                            typename Matrix :: DeviceType,
                                                            typename Matrix :: IndexType> >
class tnlSORSolver : public tnlObject,
                     public tnlIterativeSolver< typename Matrix :: RealType,
                                                typename Matrix :: IndexType >
{

   typedef typename Matrix :: RealType RealType;
   typedef typename Matrix :: IndexType IndexType;
   typedef typename Matrix :: DeviceType DeviceType;
   typedef Matrix MatrixType;
   typedef Preconditioner PreconditionerType;

   public:

   tnlSORSolver();

   tnlString getType() const;


   void setOmega( const RealType& omega );

   const RealType& getOmega() const;

   void setMatrix( const MatrixType& matrix );

   void setPreconditioner( const Preconditioner& preconditioner );

#ifdef HAVE_NOT_CXX11
   template< typename Vector,
             typename ResidueGetter >
   bool solve( const Vector& b, Vector& x );
#else
   template< typename Vector,
             typename ResidueGetter = tnlLinearResidueGetter< Matrix, Vector > >
   bool solve( const Vector& b, Vector& x );
#endif   

   ~tnlSORSolver();

   protected:

   RealType omega;

   const MatrixType* matrix;

   const PreconditionerType* preconditioner;

};

#include <implementation/solvers/linear/stationary/tnlSORSolver_impl.h>

#include <matrices/tnlCSRMatrix.h>
#include <matrices/tnlEllpackMatrix.h>
#include <matrices/tnlMultidiagonalMatrix.h>


extern template class tnlSORSolver< tnlCSRMatrix< float,  tnlHost, int > >;
extern template class tnlSORSolver< tnlCSRMatrix< double, tnlHost, int > >;
extern template class tnlSORSolver< tnlCSRMatrix< float,  tnlHost, long int > >;
extern template class tnlSORSolver< tnlCSRMatrix< double, tnlHost, long int > >;

/*extern template class tnlSORSolver< tnlEllpackMatrix< float,  tnlHost, int > >;
extern template class tnlSORSolver< tnlEllpackMatrix< double, tnlHost, int > >;
extern template class tnlSORSolver< tnlEllpackMatrix< float,  tnlHost, long int > >;
extern template class tnlSORSolver< tnlEllpackMatrix< double, tnlHost, long int > >;

extern template class tnlSORSolver< tnlMultiDiagonalMatrix< float,  tnlHost, int > >;
extern template class tnlSORSolver< tnlMultiDiagonalMatrix< double, tnlHost, int > >;
extern template class tnlSORSolver< tnlMultiDiagonalMatrix< float,  tnlHost, long int > >;
extern template class tnlSORSolver< tnlMultiDiagonalMatrix< double, tnlHost, long int > >;*/


#ifdef HAVE_CUDA
extern template class tnlSORSolver< tnlCSRMatrix< float,  tnlCuda, int > >;
extern template class tnlSORSolver< tnlCSRMatrix< double, tnlCuda, int > >;
extern template class tnlSORSolver< tnlCSRMatrix< float,  tnlCuda, long int > >;
extern template class tnlSORSolver< tnlCSRMatrix< double, tnlCuda, long int > >;

// TODO: fix this
/*
extern template class tnlSORSolver< tnlEllpackMatrix< float,  tnlCuda, int > >;
extern template class tnlSORSolver< tnlEllpackMatrix< double, tnlCuda, int > >;
extern template class tnlSORSolver< tnlEllpackMatrix< float,  tnlCuda, long int > >;
extern template class tnlSORSolver< tnlEllpackMatrix< double, tnlCuda, long int > >;
*/

/*
extern template class tnlSORSolver< tnlMutliDiagonalMatrix< float,  tnlCuda, int > >;
extern template class tnlSORSolver< tnlMutliDiagonalMatrix< double, tnlCuda, int > >;
extern template class tnlSORSolver< tnlMutliDiagonalMatrix< float,  tnlCuda, long int > >;
extern template class tnlSORSolver< tnlMutliDiagonalMatrix< double, tnlCuda, long int > >;
*/
#endif

#endif
