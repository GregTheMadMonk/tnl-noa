/***************************************************************************
                          BICGStab.h  -  description
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
class BICGStab
: public LinearSolver< Matrix >
{
   using Base = LinearSolver< Matrix >;
public:
   using RealType = typename Base::RealType;
   using DeviceType = typename Base::DeviceType;
   using IndexType = typename Base::IndexType;
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" ) override;

   bool solve( ConstVectorViewType b, VectorViewType x ) override;

protected:
   void compute_residue( VectorViewType r, ConstVectorViewType x, ConstVectorViewType b );

   void preconditioned_matvec( ConstVectorViewType src, VectorViewType dst );

   void setSize( const VectorViewType& x );

   bool exact_residue = false;

   typename Traits< Matrix >::VectorType r, r_ast, p, s, Ap, As, M_tmp;
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include "BICGStab.hpp"
