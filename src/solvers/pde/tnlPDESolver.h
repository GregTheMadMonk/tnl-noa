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

template< typename Problem,
          typename TimeStepper >
class tnlPDESolver : public tnlObject
{
   public:

   typedef typename TimeStepper :: RealType RealType;
   typedef typename TimeStepper :: DeviceType DeviceType;
   typedef typename TimeStepper :: IndexType IndexType;
   typedef typename TimeStepper :: ProblemType ProblemType;
   
   tnlPDESolver();

   void setTimeStepper( TimeStepper& timeStepper );

   void setProblem( ProblemType& problem );

   bool setFinalTime( const RealType& finalT );

   const RealType& getFinalTine() const;

   bool setSnapshotTau( const RealType& tau );
   
   const RealType& getSnapshotTau() const;

   bool solve();

   protected:

   TimeStepper* timeStepper;

   RealType finalTime, snapshotTau;

   ProblemType* problem;

};

#include <implementation/solvers/pde/tnlPDESolver_impl.h>

#endif /* TNLPDESOLVER_H_ */
