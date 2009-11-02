/***************************************************************************
                          mfilename.h  -  description
                             -------------------
    begin                : 2007/06/18
    copyright            : (C) 2007 by Tomas Oberhuber
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

#ifndef mfilenameH
#define mfilenameH

class mString;

void FileNameBaseNumberEnding( const char* base_name,
                               int number,
                               int index_size,
                               const char* ending,
                               mString& file_name );

void RemoveFileExtension( mString& file_name );

#endif
