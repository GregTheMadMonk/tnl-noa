/***************************************************************************
                          mLogger.cpp  -  description
                             -------------------
    begin                : 2007/08/22
    copyright            : (C) 2007 by Tomá¹ Oberhuber
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

#include <iomanip>
#include "mLogger.h"

//--------------------------------------------------------------------------
void mLogger :: WriteHeader( const char* title )
{
   int fill = stream. fill(); 
   int title_length = strlen( title );
   stream << "+" << setfill( '-' ) << setw( width ) << "+" << endl;
   stream << "|" << setfill( ' ' ) << setw( width ) << "|" << endl;
   stream << "|" << setw( width / 2 + title_length / 2 )
    << title << setw( width / 2 - title_length / 2  ) << "|" << endl;
   stream << "|" << setfill( ' ' ) << setw( width ) << "|" << endl;
   stream << "+" << setfill( '-' ) << setw( width ) << "+" << endl;
   stream. fill( fill );
}
//--------------------------------------------------------------------------
void mLogger :: WriteSeparator()
{
   int fill = stream. fill(); 
   stream << "+" << setfill( '-' ) << setw( width ) << "+" << endl;
   stream. fill( fill );
}

