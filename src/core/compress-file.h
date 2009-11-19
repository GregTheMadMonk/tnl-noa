/***************************************************************************
                          compress-file.h  -  description
                             -------------------
    begin                : 2007/07/02
    copyright            : (C) 2007 by Tomá¹ Oberhuber
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

#ifndef compress_fileH
#define compress_fileH

bool CompressFile( const char* file_name, const char* format );

bool UnCompressFile( const char* file_name, const char* format );

#endif
