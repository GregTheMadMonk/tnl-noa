/***************************************************************************
                          tnlPGMImage.h  -  description
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

#ifndef TNLPGMIMAGE_H
#define	TNLPGMIMAGE_H

#include <core/tnlString.h>
#include <core/io/tnlImage.h>

template< typename Index = int >
class tnlPGMImage : public tnlImage< Index >
{
   public:
      
      typedef Index IndexType;
      
      tnlPGMImage();
       
      bool open( const tnlString& fileName );
      
      protected:
         
         bool binary;
         
         IndexType colors;
      
    
};

#include <core/io/tnlPGMImage_impl.h>

#endif	/* TNLPGMIMAGE_H */

