// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

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
class TimeIndependentPDESolver : public PDESolver< typename Problem::RealType, typename Problem::IndexType >
{
public:
   using RealType = typename Problem::RealType;
   using DeviceType = typename Problem::DeviceType;
   using IndexType = typename Problem::IndexType;
   using BaseType = PDESolver< RealType, IndexType >;
   using ProblemType = Problem;
   using MeshType = typename ProblemType::MeshType;
   using DofVectorType = typename ProblemType::DofVectorType;
   using MeshPointer = Pointers::SharedPointer< MeshType, DeviceType >;
   using DofVectorPointer = Pointers::SharedPointer< DofVectorType, DeviceType >;
   using CommonDataType = typename ProblemType::CommonDataType;
   using CommonDataPointer = typename ProblemType::CommonDataPointer;

   TimeIndependentPDESolver();

   static void
   configSetup( Config::ConfigDescription& config, const String& prefix = "" );

   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" );

   bool
   writeProlog( Logger& logger, const Config::ParameterContainer& parameters );

   void
   setProblem( ProblemType& problem );

   bool
   solve();

   bool
   writeEpilog( Logger& logger ) const;

protected:
   MeshPointer meshPointer;

   Pointers::SharedPointer< Meshes::DistributedMeshes::DistributedMesh< MeshType > > distributedMeshPointer;

   CommonDataPointer commonDataPointer;

   DofVectorPointer dofs;

   ProblemType* problem;
};

}  // namespace PDE
}  // namespace Solvers
}  // namespace TNL

#include <TNL/Solvers/PDE/TimeIndependentPDESolver.hpp>
