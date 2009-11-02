/***************************************************************************
                          fmm2d.h  -  description
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

#ifndef fmm2dH
#define fmm2dH

#include <mdiff.h>

void DstFastMarching2D( mGrid2D< double >& phi, 
                        const double& band_width,
                        const double& delta );

#endif
