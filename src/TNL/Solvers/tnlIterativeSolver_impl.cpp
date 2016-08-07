/***************************************************************************
                          tnlIterativeSolver_impl.cpp  -  description
                             -------------------
    begin                : Mar 17, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Solvers/tnlIterativeSolver.h>

namespace TNL {

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
} // namespace TNL