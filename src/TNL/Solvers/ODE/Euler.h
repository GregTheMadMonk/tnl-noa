/***************************************************************************
                          Euler.h  -  description
                             -------------------
    begin                : 2008/04/01
    copyright            : (C) 2008 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <math.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Solvers/ODE/ExplicitSolver.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Timer.h>

namespace TNL {
namespace Solvers {
namespace ODE {

template< typename Problem >
class Euler : public ExplicitSolver< Problem >
{
   public:

      using ProblemType = Problem;
      using DofVectorType = typename ProblemType::DofVectorType;
      using RealType = typename ProblemType::RealType;
      using DeviceType = typename ProblemType::DeviceType;
      using IndexType  = typename ProblemType::IndexType;
      using DofVectorView = typename ViewTypeGetter< DofVectorType >::Type;
      using DofVectorPointer = Pointers::SharedPointer<  DofVectorType, DeviceType >;

      Euler();

      static String getType();

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" );

      bool setup( const Config::ParameterContainer& parameters,
                 const String& prefix = "" );

      void setCFLCondition( const RealType& cfl );

      const RealType& getCFLCondition() const;

      bool solve( DofVectorPointer& u );

   protected:
      DofVectorPointer _k1;

      RealType cflCondition;
};

} // namespace ODE
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/ODE/Euler.hpp>
