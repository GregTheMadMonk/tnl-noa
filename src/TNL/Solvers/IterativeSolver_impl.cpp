/***************************************************************************
                          IterativeSolver_impl.cpp  -  description
                             -------------------
    begin                : Mar 17, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Solvers/IterativeSolver.h>

namespace TNL {
namespace Solvers {   

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

template class IterativeSolver< float,  int >;
template class IterativeSolver< double, int >;
template class IterativeSolver< float,  long int >;
template class IterativeSolver< double, long int >;

#ifdef HAVE_CUDA
template class IterativeSolver< float,  int >;
template class IterativeSolver< double, int >;
template class IterativeSolver< float,  long int >;
template class IterativeSolver< double, long int >;
#endif

#endif
} // namespace Solvers
} // namespace TNL