/***************************************************************************
                          init_band.h  -  description
                             -------------------
    begin                : 2005/08/10
    copyright            : (C) 2005 by Tom� Oberhuber
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

#ifndef init_bandH
#define init_bandH

#include <math.h>

#include <diff/mGrid2D.h>
#include <core/mVector.h>
#include <core/mfuncs.h>

//! Cretes initial band along the curve given as a zero level set
/*! Only points having a neighbour with different sign will be
    in the initial band.
 */
void InitBand( const mGrid2D< double >& phi,
               mVector< 2, int >* stack,
               int& stack_end,
               mField2D< char >& in_stack );

//! Compute distance function in the initial band
/*! This function just approximate zero level set curve
    and compute distance from it.
 */
void RedistanceBand( mGrid2D< double >& phi,
                     mVector< 2, int >* stack,
                     int stack_end,
                     mField2D< char >& in_stack );

#endif
