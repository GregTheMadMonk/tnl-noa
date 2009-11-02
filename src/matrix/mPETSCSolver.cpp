/***************************************************************************
                          mPETSCSolver.cpp  -  description
                             -------------------
    begin                : 2008/05/12
    copyright            : (C) 2008 by Tomá¹ Oberhuber
    email                : oberhuber@seznam.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "mPETSCSolver.h"

//--------------------------------------------------------------------------
#ifdef HAVE_PETSC
/*PetscErrorCode PETSCSolverMonitorCallback( KSP petsc_solver, PetscInt iter, PetscReal rnorm, void* ctx )
{
   cout << "*" << flush;
}*/
#endif

