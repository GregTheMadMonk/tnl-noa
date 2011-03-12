/***************************************************************************
                          tnlDebugStructure.h  -  description
                             -------------------
    begin                : 2005/02/20
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

#ifndef tnlDebugStructureH
#define tnlDebugStructureH

#include <list>
#include "tnlDebugEntry.h"
#include "tnlDebugGroup.h"

class tnlDebugStructure
{
   //! List of groups - classes or function groups
   list< tnlDebugGroup* > debug_groups;

   //! List of groups - classes or function groups
   list< tnlDebugEntry* > alone_entries;

   public:

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
   tnlDebugStructure();

   //! Destructor
   ~tnlDebugStructure();

   //! This turns the debug mode on
   /*!**
    * This is usually done during the parsing of the debug file.
    */
   void setDebug( bool debug );

   //! Adds debug group to the list
   void AppendGroup( tnlDebugGroup* group );

   //! Adds stand alone entry - with no class and no group
   void AppendAloneEntry( tnlDebugEntry* entry );

   //! Check entry whether it is set for debug
   bool Debug( const char* group_name,
               const char* function_name );

   //! Check entry whether it is set to allow interactive debuging
   bool Interactive( const char* group_name,
                     const char* function_name );
   
   void Print();
   
};


#endif
