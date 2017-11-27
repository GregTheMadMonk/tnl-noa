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

#include <core/tnlObject.h>
#include <config/tnlConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <solvers/tnlSolverMonitor.h>
#include <core/tnlLogger.h>

template< typename Problem >
class tnlTimeIndependentPDESolver : public tnlObject
{
   public:

      typedef Problem ProblemType;
      typedef typename ProblemType::RealType RealType;
      typedef typename ProblemType::DeviceType DeviceType;
      typedef typename ProblemType::IndexType IndexType;
      typedef typename ProblemType::MeshType MeshType;
      typedef typename ProblemType::DofVectorType DofVectorType;
      typedef typename ProblemType::MeshDependentDataType MeshDependentDataType;      

      tnlTimeIndependentPDESolver();

      static void configSetup( tnlConfigDescription& config,
                               const String& prefix = "" );

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" );

      bool writeProlog( tnlLogger& logger,
                        const Config::ParameterContainer& parameters );


      void setProblem( ProblemType& problem );

      void setComputeTimer( tnlTimer& computeTimer );
      
      void setIoTimer( tnlTimer& ioTimer );

      bool solve();

      bool writeEpilog( tnlLogger& logger ) const;

   protected:

      MeshType mesh;

      DofVectorType dofs;

      MeshDependentDataType meshDependentData;

      ProblemType* problem;

      tnlTimer *computeTimer;
};

#include <solvers/pde/tnlTimeIndependentPDESolver_impl.h>

