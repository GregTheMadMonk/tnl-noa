/***************************************************************************
                          tnlPDESolver.h  -  description
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

#ifndef TNLPDESOLVER_H_
#define TNLPDESOLVER_H_

#include <core/tnlObject.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>
#include <solvers/tnlSolverMonitor.h>

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

   void setIoRtTimer( tnlTimerRT& ioRtTimer);

   void setComputeRtTimer( tnlTimerRT& computeRtTimer );

   void setIoCpuTimer( tnlTimerCPU& ioCpuTimer );

   void setComputeCpuTimer( tnlTimerCPU& computeCpuTimer );

   bool solve();

   protected:

   MeshType mesh;

   DofVectorType dofs;

   DofVectorType auxiliaryDofs;

   TimeStepper* timeStepper;

   RealType initialTime, finalTime, snapshotPeriod, timeStep, timeStepOrder;

   ProblemType* problem;

   tnlTimerRT *ioRtTimer, *computeRtTimer;

   tnlTimerCPU *ioCpuTimer, *computeCpuTimer;

};

#include <solvers/pde/tnlPDESolver_impl.h>

#endif /* TNLPDESOLVER_H_ */
