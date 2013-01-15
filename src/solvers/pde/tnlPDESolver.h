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

template< typename TimeStepper >
class tnlPDESolver
{
   public:

   void setTimeStepper( TimeStepper& timeStepper );

   protected:

   TimeStepper* timeStepper;

};


#endif /* TNLPDESOLVER_H_ */
