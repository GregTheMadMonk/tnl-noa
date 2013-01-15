/***************************************************************************
                          tnlExplicitTimeStepper.h  -  description
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

#ifndef TNLEXPLICITTIMESTEPPER_H_
#define TNLEXPLICITTIMESTEPPER_H_

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
class tnlExplicitTimeStepper
{
   public:

   typedef Problem ProblemType;
   typedef OdeSolver< ProblemType > OdeSolverType;

   void setSolver( OdeSolverType& odeSolver );

   protected:

   OdeSolverType* odeSolver;
};



#endif /* TNLEXPLICITTIMESTEPPER_H_ */
