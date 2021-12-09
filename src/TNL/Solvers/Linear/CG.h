/***************************************************************************
                          CG.h  -  description
                             -------------------
    begin                : 2007/07/31
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
class CG
: public LinearSolver< Matrix >
{
   using Base = LinearSolver< Matrix >;
public:
   using RealType = typename Base::RealType;
   using DeviceType = typename Base::DeviceType;
   using IndexType = typename Base::IndexType;
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;
   using VectorType = typename Traits< Matrix >::VectorType;

   bool solve( ConstVectorViewType b, VectorViewType x ) override;

protected:
   void setSize( const VectorViewType& x );

   VectorType r, p, Ap, z;
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include "CG.hpp"
