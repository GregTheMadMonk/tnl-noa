/***************************************************************************
                          mPETSCPreconditioner.h  -  description
                             -------------------
    begin                : 2008/05/13
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

#ifndef mPETSCPreconditionerH
#define mPETSCPreconditionerH

#include <matrix/mPreconditioner.h>

#ifdef HAVE_PETSC
#include <petscksp.h>
#else
struct PC;
#endif

#include <iostream>

using namespace std;

//! This class wraps PETSc preconditioners
template< typename T > class mPETSCPreconditioner : public mPreconditioner< T >
{
#ifdef HAVE_PETSC
   PC petsc_preconditioner;
#endif

   public:

   mPETSCPreconditioner( const char* type )
   {
#ifdef HAVE_PETSC
      PCCreate( MPI_COMM_SELF, &petsc_preconditioner );
      if( strcmp( type, "none" ) == 0 )
         PCSetType( petsc_preconditioner, PCNONE );
      if( strcmp( type, "lu" ) == 0 )
         PCSetType( petsc_preconditioner, PCLU );
      if( strcmp( type, "ilu" ) == 0 )
         PCSetType( petsc_preconditioner, PCILU );
#else
      cerr << "Missing support for PETSC at the file " << __FILE__ << " line " << __LINE__ << endl;
#endif
   };
   
   void GetData( PC& precond ) const
   {
#ifdef HAVE_PETSC
      precond = petsc_preconditioner;
#else
      cerr << "Missing support for PETSC at the file " << __FILE__ << " line " << __LINE__ << endl;
#endif
   };
   
   bool Solve( const T* b, T* x ) const
   {
      cerr << "Do not call mPETSCPreconditioner :: Solve. Connect it to mPETSCSolver using mPETSCPreconditioner :: GetData( PC& precond )." << endl;
      abort();
   };

   ~mPETSCPreconditioner()
   {
#ifdef HAVE_PETSC
      PCDestroy( petsc_preconditioner );
#else
      cerr << "Missing support for PETSC at the file " << __FILE__ << " line " << __LINE__ << endl;
#endif
   };
};

#endif
