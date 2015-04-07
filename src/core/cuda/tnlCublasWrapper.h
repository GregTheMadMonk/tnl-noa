/***************************************************************************
                          tnlCublasWrapper.h  -  description
                             -------------------
    begin                : Apr 7, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
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

#ifndef TNLCUBLASWARPER_H
#define	TNLCUBLASWARPER_H

template< typename Real1, 
          typename Real2,
          typename Index >
class tnlCublasWrapper
{
    public:
        static bool sdot( const Real1* v1, const Real2* v2, const Index size, Real1& result)
        {
            return false;
        }
};

#endif	/* TNLCUBLASWARPER_H */

