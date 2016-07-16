/***************************************************************************
                          tnlDebugGroup.cpp  -  description
                             -------------------
    begin                : 2005/08/16
    copyright            : (C) 2005 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

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
