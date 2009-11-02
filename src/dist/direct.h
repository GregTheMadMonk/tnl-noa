/***************************************************************************
                          direct.h  -  description
                             -------------------
    begin                : 2005/08/10
    copyright            : (C) 2005 by Tomá¹ Oberhuber
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

#ifndef directH
#define directH

#include <mdiff.h>

enum mDstDirection { mDstSouth,
                     mDstNorth,
                     mDstEast,
                     mDstWest };

enum mDstState { mDstFixed, mDstTentative, mDstFar };
                     

double UpdatePoint2D( const mGrid2D< double >& f,
                      int i, int j,
                      const double& smallest,
                      mDstDirection smallest_direction,
                      mDstState e,
                      mDstState w,
                      mDstState s,
                      mDstState n );

#endif
