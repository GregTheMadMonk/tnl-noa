/***************************************************************************
                          fsm2d.h  -  description
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

#ifndef fsm2dH
#define fsm2dH

#include <mdiff.h>

#define MDST_VERBOSE_ON true
#define MDST_VERBOSE_OFF false

bool DstFastSweeping2D( mGrid2D< double >& phi, 
                        int sweepings = 1,
                        mField2D< bool >* _fixed = 0,
                        mGrid2D< double >* extend_velocity = 0,
                        bool verbose = false );

#endif
