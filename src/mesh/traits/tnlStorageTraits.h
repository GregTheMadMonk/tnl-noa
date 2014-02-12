/***************************************************************************
                          tnlStorageTraits.h  -  description
                             -------------------
    begin                : Feb 11, 2014
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

#ifndef TNLSTORAGETRAITS_H_
#define TNLSTORAGETRAITS_H_

template< bool storageEnabled >
class tnlStorageTraits
{
   public:

   enum { enabled = storageEnabled };
};


#endif /* TNLSTORAGETRAITS_H_ */
