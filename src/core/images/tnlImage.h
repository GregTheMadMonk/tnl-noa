/***************************************************************************
                          tnlImage.h  -  description
                             -------------------
    begin                : Jul 20, 2015
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

#ifndef TNLIMAGE_H
#define	TNLIMAGE_H

template< typename Index = int >
class tnlImage
{
   public:
      
      typedef Index IndexType;
      
      tnlImage() : width( 0 ), height( 0 ) {};
      
      IndexType getWidth() const
      {
         return this->width;
      }
      
      IndexType getHeight() const
      {
         return this->height;
      }
      
   protected:
      
      IndexType width, height;
      
};


#endif	/* TNLIMAGE_H */

