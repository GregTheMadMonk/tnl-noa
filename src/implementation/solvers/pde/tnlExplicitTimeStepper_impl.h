/***************************************************************************
                          tnlExplicitTimeStepper_impl.h  -  description
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

#ifndef TNLEXPLICITTIMESTEPPER_IMPL_H_
#define TNLEXPLICITTIMESTEPPER_IMPL_H_

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
tnlExplicitTimeStepper< Problem, OdeSolver > :: tnlExplicitTimeStepper()
: odeSolver( 0 )
{
};

template< typename Problem,
          template < typename OdeProblem > class OdeSolver >
void tnlExplicitTimeStepper< Problem, OdeSolver > :: setSolver( OdeSolver< Problem >& odeSolver )
{
   this -> odeSolver = &odeSolver;
};



#endif /* TNLEXPLICITTIMESTEPPER_IMPL_H_ */
