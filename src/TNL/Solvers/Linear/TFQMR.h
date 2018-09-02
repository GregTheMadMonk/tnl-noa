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

#include <TNL/Containers/Vector.h>
#include <TNL/Solvers/Linear/Preconditioners/Dummy.h>
#include <TNL/Solvers/IterativeSolver.h>

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix,
          typename Preconditioner = Preconditioners::Dummy< typename Matrix :: RealType,
                                                            typename Matrix :: DeviceType,
                                                            typename Matrix :: IndexType> >
class TFQMR
: public LinearSolver< Matrix, Preconditioner >,
  public IterativeSolver< typename Matrix::RealType,
                          typename Matrix::IndexType >
{
   using Base = LinearSolver< Matrix, Preconditioner >;
public:
   using RealType = typename Base::RealType;
   using DeviceType = typename Base::DeviceType;
   using IndexType = typename Base::IndexType;
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;

   TFQMR();

   String getType() const;

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" );

   bool solve( ConstVectorViewType b, VectorViewType x ) override;

protected:
   void setSize( IndexType size );

   Containers::Vector< RealType, DeviceType, IndexType > d, r, w, u, v, r_ast, Au, M_tmp;

   IndexType size;
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/TFQMR_impl.h>
