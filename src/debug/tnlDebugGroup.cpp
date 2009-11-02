/***************************************************************************
                          tnlDebugGroup.cpp  -  description
                             -------------------
    begin                : 2005/08/16
    copyright            : (C) 2005 by Tomá¹ Oberhuber
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

#include "tnlDebugGroup.h"

//--------------------------------------------------------------------------
tnlDebugGroup :: tnlDebugGroup()
   : debug( true ),
     interactive( true ),
     default_debug( false ),
     default_interactive( false )
{
}
//--------------------------------------------------------------------------
tnlDebugGroup :: ~tnlDebugGroup()
{
   list< tnlDebugEntry* > :: iterator it = debug_entries. begin();
   while( it != debug_entries. end() ) delete * it ++;
}
//--------------------------------------------------------------------------
