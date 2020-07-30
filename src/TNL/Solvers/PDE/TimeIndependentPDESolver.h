/***************************************************************************
                          tnlTimeIndependentPDESolver.h  -  description
                             -------------------
    begin                : Jan 15, 2013
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

#pragma once

#include <TNL/Logger.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Solvers/PDE/PDESolver.h>
#include <TNL/Problems/CommonData.h>
#include <TNL/Pointers/SharedPointer.h>

#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>

namespace TNL {
namespace Solvers {
namespace PDE {

template< typename Problem >
class TimeIndependentPDESolver : public PDESolver< typename Problem::RealType,
                                                   typename Problem::IndexType >
{
   public:

      using RealType = typename Problem::RealType;
      using DeviceType = typename Problem::DeviceType;
      using IndexType = typename Problem::IndexType;
      using BaseType = PDESolver< RealType, IndexType >;
      using ProblemType = Problem;
      typedef typename ProblemType::MeshType MeshType;
      typedef typename ProblemType::DofVectorType DofVectorType;
      typedef Pointers::SharedPointer< MeshType, DeviceType > MeshPointer;
      typedef Pointers::SharedPointer< DofVectorType, DeviceType > DofVectorPointer;
      typedef typename ProblemType::CommonDataType CommonDataType;
      typedef typename ProblemType::CommonDataPointer CommonDataPointer;


      TimeIndependentPDESolver();

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" );

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" );

      bool writeProlog( Logger& logger,
                        const Config::ParameterContainer& parameters );


      void setProblem( ProblemType& problem );

      bool solve();

      bool writeEpilog( Logger& logger ) const;

   protected:

      MeshPointer meshPointer;

      Meshes::DistributedMeshes::DistributedMesh<MeshType> distributedMesh;

      CommonDataPointer commonDataPointer;

      DofVectorPointer dofs;

      ProblemType* problem;
};

} // namespace PDE
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/PDE/TimeIndependentPDESolver_impl.h>
