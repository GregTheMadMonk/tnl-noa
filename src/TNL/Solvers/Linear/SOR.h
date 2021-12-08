/***************************************************************************
                          SOR.h  -  description
                             -------------------
    begin                : 2007/07/30
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
class SOR
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

   void setOmega( const RealType& omega );

   const RealType& getOmega() const;

   bool solve( ConstVectorViewType b, VectorViewType x ) override;

protected:
   RealType omega = 1.0;
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include "SOR.hpp"
