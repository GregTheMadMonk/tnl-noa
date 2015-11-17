/***************************************************************************
                          tnlGridEntityGetter.h  -  description
                             -------------------
    begin                : Nov 15, 2015
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

#ifndef TNLGRIDENTITYGETTER_H
#define TNLGRIDENTITYGETTER_H

template< typename Grid,
          int EntityDimensions >
class tnlGridEntityGetter
{
   //static_assert( false, "Wrong mesh type or entity topology." );
};

/***
 * The main code is in template specializations in tnlGridEntityIndexer.h 
 */

#endif	/* TNLGRIDENTITYGETTER_H */

