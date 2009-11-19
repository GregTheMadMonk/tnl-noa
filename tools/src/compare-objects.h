/***************************************************************************
                          compare-objets.h  -  description
                             -------------------
    begin                : 2009/08/14
    copyright            : (C) 2009 by Tomá¹ Oberhuber
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

#ifndef compare_objetsH
#define compare_objetsH

#include <diff/mGrid2D.h>
#include <diff/mGrid3D.h>

bool Compare( const mGrid2D< double >& u1,
              const mGrid2D< double >& u2,
              double& l1_norm,
              double& l2_norm,
              double& max_norm,
              mGrid2D< double >& difference );

bool Compare( const mGrid3D< double >& u1,
              const mGrid3D< double >& u2,
              double& l1_norm,
              double& l2_norm,
              double& max_norm,
              mGrid3D< double >& difference );

#endif
