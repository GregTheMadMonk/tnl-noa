/***************************************************************************
                          tnlExplicitSolver.h  -  description
                             -------------------
    begin                : 2007/06/17
    copyright            : (C) 2007 by Tomas Oberhuber
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

#ifndef tnlExplicitSolverH
#define tnlExplicitSolverH

#include <iomanip>
#include <core/tnlTimerCPU.h>
#include <core/tnlTimerRT.h>
#include <core/tnlFlopsCounter.h>
#include <core/tnlObject.h>
#include <solvers/ode/tnlODESolverMonitor.h>
#include <solvers/tnlIterativeSolver.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>

template< class Problem >
class tnlExplicitSolver : public tnlIterativeSolver< typename Problem::RealType,
                                                     typename Problem::IndexType >
{
   public:
   
   typedef Problem ProblemType;
   typedef typename Problem :: DofVectorType DofVectorType;
   typedef typename Problem :: RealType RealType;
   typedef typename Problem :: DeviceType DeviceType;
   typedef typename Problem :: IndexType IndexType;

   tnlExplicitSolver();

   static void configSetup( tnlConfigDescription& config,
                            const tnlString& prefix = "" );

   bool setup( const tnlParameterContainer& parameters,
              const tnlString& prefix = "" );

   void setProblem( Problem& problem );

   void setTime( const RealType& t );

   const RealType& getTime() const;
   
   void setStopTime( const RealType& stopTime );

   RealType getStopTime() const;

   void setTau( const RealType& tau );
   
   const RealType& getTau() const;

   void setMaxTau( const RealType& maxTau );

   const RealType& getMaxTau() const;

   void setMPIComm( MPI_Comm comm );
  
   void setVerbose( IndexType v );

   void setTimerCPU( tnlTimerCPU* timer );

   void setTimerRT( tnlTimerRT* timer );
   
   virtual bool solve( DofVectorType& u ) = 0;

   void setTestingMode( bool testingMode );

   void setRefreshRate( const IndexType& refreshRate );

   void setSolverMonitor( tnlODESolverMonitor< RealType, IndexType >& solverMonitor );

   void refreshSolverMonitor();

protected:
    
   /****
    * Current time of the parabolic problem.
    */
   RealType time;

   /****
    * The method solve will stop when reaching the stopTime.
    */
   RealType stopTime;

   /****
    * Current time step.
    */
   RealType tau;

   RealType maxTau;

   MPI_Comm solver_comm;

   IndexType verbosity;

   tnlTimerCPU* cpu_timer;
   
   tnlTimerRT* rt_timer;

   bool testingMode;

   Problem* problem;

   tnlODESolverMonitor< RealType, IndexType >* solverMonitor;

   /****
    * Auxiliary array for the computation of the solver residue on CUDA device.
    */
   tnlVector< RealType, DeviceType, IndexType > cudaBlockResidue;
};

#include <implementation/solvers/ode/tnlExplicitSolver_impl.h>


#endif
