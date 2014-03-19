/***************************************************************************
                          tnlDynamicTypeTag.h  -  description
                             -------------------
    begin                : Mar 13, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLDYNAMICTYPETAG_H_
#define TNLDYNAMICTYPETAG_H_

template< typename Element >
struct tnlDynamicTypeTag
{
   enum { value = false };
};


#endif /* TNLDYNAMICTYPETAG_H_ */
