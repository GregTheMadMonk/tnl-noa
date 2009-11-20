/***************************************************************************
                          mPETSCSolver.h  -  description
                             -------------------
    begin                : 2008/05/09
    copyright            : (C) 2008 by Tomá¹ Oberhuber
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

#ifndef mPETSCSolverH
#define mPETSCSolverH

#ifdef HAVE_PETSC
#include <petscksp.h>
#include <petsc.h>
#endif

#include <matrix/mMatrixSolver.h>

#ifdef HAVE_PETSC
template< typename T > inline PetscErrorCode PETSCSolverMonitorCallback( KSP petsc_solver, PetscInt iter, PetscReal rnorm, void* ctx );
#endif

//! This class is a wrapper for the PETSc solvers.
template< typename T > class mPETSCSolver : public mMatrixSolver< T >
{
#ifdef HAVE_PETSC
   KSP petsc_solver;
#endif

   long int gmre_restarting;

   public:

   mPETSCSolver( const char* solver_name )
   {
#ifdef HAVE_PETSC
      PetscErrorCode ierr = KSPCreate( MPI_COMM_SELF, &petsc_solver );
      if( strcmp( solver_name, "cg" ) == 0 )
         KSPSetType( petsc_solver, KSPCG );
      if( strcmp( solver_name, "bicg" ) == 0 )
         KSPSetType( petsc_solver, KSPBICG );

      if( strcmp( solver_name, "gmres" ) == 0 )
         KSPSetType( petsc_solver, KSPGMRES );
      else assert( 0 );
#else
      cerr << "Missing support for PETSC at the file " << __FILE__ << " line " << __LINE__ << endl;
#endif
   };

   void SetRestarting( long int rest )
   {
#ifdef HAVE_PETSC
      KSPGMRESSetRestart( petsc_solver, rest );
      gmre_restarting = rest;
#else
      cerr << "Missing support for PETSC at the file " << __FILE__ << " line " << __LINE__ << endl;
#endif
   };
   
   void PrintOut()
   {
#ifdef HAVE_PETSC
      PetscInt it;
      KSPGetIterationNumber( petsc_solver, &it );
      PetscReal res;
      KSPGetResidualNorm( petsc_solver, &res );
      mMatrixSolver< T > :: iteration = it;
      mMatrixSolver< T > :: residue = res;
      mMatrixSolver< T > :: PrintOut();
#else
      cerr << "Missing support for PETSC at the file " << __FILE__ << " line " << __LINE__ << endl;
#endif
   };

   bool Solve( const tnlBaseMatrix< T >& A,
               const T* b,
               T* x, 
               const double& max_residue,
               const long int max_iterations,
               mPreconditioner< T >* precond = 0 )
   {
#ifdef HAVE_PETSC
      assert( A. GetMatrixClass() == mMatrixClass :: petsc );
      Vec petsc_x, petsc_b;
      mPETSCMatrix< T >* petsc_matrix = ( mPETSCMatrix< T >* ) & A;
      Mat matrix;
      petsc_matrix -> GetData( matrix );

      MatAssemblyBegin( matrix, MAT_FINAL_ASSEMBLY );
      MatAssemblyEnd( matrix, MAT_FINAL_ASSEMBLY );
      
      const long int size = petsc_matrix -> GetSize();

      VecCreateSeqWithArray( MPI_COMM_SELF, size, x, &petsc_x );
      VecCreateSeqWithArray( MPI_COMM_SELF, size, b, &petsc_b );
      
      T normb;
      long int i;
      for( i = 0; i < size; i ++ )
      {
         normb += b[ i ] * b[ i ];
      }
      normb = sqrt( normb );
      
      KSPGMRESSetCGSRefinementType( petsc_solver, KSP_GMRES_CGS_REFINE_IFNEEDED );
      
      KSPMonitorSet( petsc_solver, PETSCSolverMonitorCallback< T >, this, PETSC_NULL );

      KSPSetTolerances( petsc_solver, max_residue, PETSC_DEFAULT, PETSC_DEFAULT, max_iterations );
      KSPSetOperators( petsc_solver, matrix, matrix, SAME_NONZERO_PATTERN);
      KSPSetInitialGuessNonzero( petsc_solver, PETSC_TRUE );

      PC pc;
      KSPGetPC( petsc_solver, &pc );
      PCSetType( pc, PCNONE );
      //PCFactorSetUseDropTolerance( pc, 1.0e-10, 0.1, 100 );
      PCFactorSetShiftPd( pc, PETSC_TRUE );

      KSPSolve( petsc_solver, petsc_b, petsc_x); 

      VecDestroy( petsc_x );
      VecDestroy( petsc_b );

      PetscInt its;
      KSPGetIterationNumber( petsc_solver, &its );
      mMatrixSolver< T > :: iteration = its;
      
      if( mMatrixSolver< T > :: iteration < 0 ) return false;
      
      PetscReal res;
      KSPGetResidualNorm( petsc_solver, &res );
      mMatrixSolver< T > :: residue = res / normb;
      
      //KSPDestroy( petsc_solver );
      
      return true;
#else
      cerr << "Missing support for PETSC at the file " << __FILE__ << " line " << __LINE__ << endl;
      return false;
#endif

   };

   ~mPETSCSolver()
   {
#ifdef HAVE_PETSC
      KSPDestroy( petsc_solver );
#else
      cerr << "Missing support for PETSC at the file " << __FILE__ << " line " << __LINE__ << endl;
#endif
   };

};

#ifdef HAVE_PETSC
template< typename T > inline PetscErrorCode PETSCSolverMonitorCallback( KSP ksp_solver, PetscInt iter, PetscReal rnorm, void* ctx )
{
   mPETSCSolver< T >* petsc_solver = ( mPETSCSolver< T > * ) ctx;
   petsc_solver -> PrintOut();
      
   return 0;
}
#endif

#endif
