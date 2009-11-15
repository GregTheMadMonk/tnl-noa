/***************************************************************************
                          initial-condition.h  -  description
                             -------------------
    begin                : 2007/06/17
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : oberhuber@seznam.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef initial_conditionH
#define initial_conditionH

#include <diff/mdiff.h>

bool GetInitialCondition( const mParameterContainer& parameters,
                          mGrid2D< double >*& u );
struct CircleData
{
   double x_pos;
   double y_pos;
   double radius;
};

struct SpiralData
{
   double radius;
   double twist;
};

#endif
