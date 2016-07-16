/***************************************************************************
                          tnlPDESolver.h  -  description
                             -------------------
    begin                : Jan 15, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLPDESOLVER_H_
#define TNLPDESOLVER_H_

#include <core/tnlObject.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>
#include <core/tnlLogger.h>

template< typename Problem,
          typename TimeStepper >
class tnlPDESolver : public tnlObject
{
   public:

      typedef typename TimeStepper::RealType RealType;
      typedef typename TimeStepper::DeviceType DeviceType;
      typedef typename TimeStepper::IndexType IndexType;
      typedef Problem ProblemType;
      typedef typename ProblemType::MeshType MeshType;
      typedef typename ProblemType::DofVectorType DofVectorType;
      typedef typename ProblemType::MeshDependentDataType MeshDependentDataType;

      tnlPDESolver();

      static void configSetup( tnlConfigDescription& config,
                               const tnlString& prefix = "" );

      bool setup( const tnlParameterContainer& parameters,
                 const tnlString& prefix = "" );

      bool writeProlog( tnlLogger& logger,
                        const tnlParameterContainer& parameters );

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

      bool writeEpilog( tnlLogger& logger ) const;

   protected:

      MeshType mesh;

      DofVectorType dofs;

      MeshDependentDataType meshDependentData;

      TimeStepper* timeStepper;

      RealType initialTime, finalTime, snapshotPeriod, timeStep, timeStepOrder;

      ProblemType* problem;

      tnlTimer *ioTimer, *computeTimer;
};

#include <solvers/pde/tnlPDESolver_impl.h>

#endif /* TNLPDESOLVER_H_ */
