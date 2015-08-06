/***************************************************************************
                          tnlIterativeSolver_impl.cpp  -  description
                             -------------------
    begin                : Mar 17, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#include <solvers/tnlIterativeSolver.h>

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

template class tnlIterativeSolver< float,  int >;
template class tnlIterativeSolver< double, int >;
template class tnlIterativeSolver< float,  long int >;
template class tnlIterativeSolver< double, long int >;

#ifdef HAVE_CUDA
template class tnlIterativeSolver< float,  int >;
template class tnlIterativeSolver< double, int >;
template class tnlIterativeSolver< float,  long int >;
template class tnlIterativeSolver< double, long int >;
#endif

#endif



