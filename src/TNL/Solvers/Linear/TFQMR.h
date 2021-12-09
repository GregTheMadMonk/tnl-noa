/***************************************************************************
                          TFQMR.h  -  description
                             -------------------
    begin                : Dec 8, 2012
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include "LinearSolver.h"

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
class TFQMR
: public LinearSolver< Matrix >
{
   using Base = LinearSolver< Matrix >;
public:
   using RealType = typename Base::RealType;
   using DeviceType = typename Base::DeviceType;
   using IndexType = typename Base::IndexType;
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;

   bool solve( ConstVectorViewType b, VectorViewType x ) override;

protected:
   void setSize( const VectorViewType& x );

   typename Traits< Matrix >::VectorType d, r, w, u, v, r_ast, Au, M_tmp;
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include "TFQMR.hpp"
