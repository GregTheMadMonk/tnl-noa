/***************************************************************************
                          PDESolver.h  -  description
                             -------------------
    begin                : Jan 15, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Object.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Logger.h>

namespace TNL {
namespace Solvers {
namespace PDE {   

template< typename Problem,
          typename TimeStepper >
class PDESolver : public Object
{
   public:

      typedef typename TimeStepper::RealType RealType;
      typedef typename TimeStepper::DeviceType DeviceType;
      typedef typename TimeStepper::IndexType IndexType;
      typedef Problem ProblemType;
      typedef typename ProblemType::MeshType MeshType;
      typedef tnlSharedPointer< MeshType, DeviceType > MeshPointer;
      typedef typename ProblemType::DofVectorType DofVectorType;
      typedef typename ProblemType::MeshDependentDataType MeshDependentDataType;
      typedef tnlSharedPointer< DofVectorType, DeviceType > DofVectorPointer;

      PDESolver();

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" );

      bool setup( const Config::ParameterContainer& parameters,
                 const String& prefix = "" );

      bool writeProlog( Logger& logger,
                        const Config::ParameterContainer& parameters );

      void setTimeStepper( TimeStepper& timeStepper );

      void setProblem( ProblemType& problem );

      void setInitialTime( const RealType& initialT );

      const RealType& getInitialTime() const;

      bool setFinalTime( const RealType& finalT );

      const RealType& getFinalTime() const;

      bool setTimeStep( const RealType& timeStep );

      const RealType& getTimeStep() const;

      bool setTimeStepOrder( const RealType& timeStepOrder );

      const RealType& getTimeStepOrder() const;

      bool setSnapshotPeriod( const RealType& period );

      const RealType& getSnapshotPeriod() const;

      void setIoTimer( tnlTimer& ioTimer);

      void setComputeTimer( tnlTimer& computeTimer );

      bool solve();

      bool writeEpilog( Logger& logger ) const;

   protected:

      MeshPointer meshPointer;

      DofVectorPointer dofsPointer;

      MeshDependentDataType meshDependentData;

      TimeStepper* timeStepper;

      RealType initialTime, finalTime, snapshotPeriod, timeStep, timeStepOrder;

      ProblemType* problem;

      tnlTimer *ioTimer, *computeTimer;
};

} // namespace PDE
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/PDE/PDESolver_impl.h>
