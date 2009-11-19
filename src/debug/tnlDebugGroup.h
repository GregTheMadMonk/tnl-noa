/***************************************************************************
                          tnlDebugGroup.h  -  description
                             -------------------
    begin                : 2005/08/16
    copyright            : (C) 2005 by Tomas Oberhuber
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

#ifndef tnlDebugGroupH
#define tnlDebugGroupH

#include <string>
#include <list>
#include "tnlDebugEntry.h"

//! This structure describes group of debug entries = functions/methods
/*! It can be either class of methods or group of functions
 */
class tnlDebugGroup
{
   public:

   //! List of functions/methods
   list< tnlDebugEntry* > debug_entries;

   //! Group name
   string group_name;

   //! Flag for debuging
   bool debug;

   //! Flag for interactive debuging
   bool interactive;

   //! Flag for default debuging
   /*! It affects functions/methods not mentioned in the
       inital file.
    */
   bool default_debug;

   //! Flag for default interactive - similar to default debug
   bool default_interactive;

   //! Basic constructor
   tnlDebugGroup();

   //! Destructor
   ~tnlDebugGroup();
};

#endif
