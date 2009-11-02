/***************************************************************************
                          tnlDebugStructure.cpp  -  description
                             -------------------
    begin                : 2004/09/05
    copyright            : (C) 2004 by Tomas Oberhuber
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

#include <string>
#include <iostream>
#include "tnlDebugStructure.h"

const char* booltostr( bool b )
{
   if( b ) return "yes";
   return "no";
}
//--------------------------------------------------------------------------
tnlDebugStructure :: tnlDebugStructure()
   : debug( true ),
     interactive( true ),
     default_debug( true ),
     default_interactive( true )
{
}
//--------------------------------------------------------------------------
tnlDebugStructure :: ~tnlDebugStructure()
{
   list< tnlDebugGroup* > :: iterator it = debug_groups. begin();
   while( it != debug_groups. end() ) delete * it ++;
}
//--------------------------------------------------------------------------
void tnlDebugStructure :: AppendGroup( tnlDebugGroup* group )
{
  //cout << "Adding group ... " << endl;
  debug_groups. push_back( group );   
}
//--------------------------------------------------------------------------
void tnlDebugStructure :: AppendAloneEntry( tnlDebugEntry* entry )
{
   //cout << "Adding entry ... " << endl;
   alone_entries. push_back( entry );
}
//--------------------------------------------------------------------------
bool tnlDebugStructure :: Debug( const char* group_name,
                                 const char* function_name )
{
   if( ! debug ) return false;
#ifdef DEBUG
   cout << "Debug: " << group_name << " :: " << function_name << endl;
#endif
   if( strlen( group_name ) != 0 )
   {
      //cout << "size = " << debug_groups. Size() << endl;
      list< tnlDebugGroup* > :: iterator it1 = debug_groups. begin();
      while( it1 != debug_groups. end() )
      {
         tnlDebugGroup* group = * it1;
         //cout << group -> group_name << endl;
         if( group -> group_name == group_name )
         {
#ifdef DEBUG
            cout << "Class " << group_name << " found." << endl;
#endif
            if( ! group -> debug ) return false;
            list< tnlDebugEntry* > :: iterator it2 = group -> debug_entries. begin();
            while( it2 != group -> debug_entries. end() )
            {
               tnlDebugEntry* entry = * it2;
               if( entry -> function_name == function_name )
                  return entry -> debug;
               it2 ++;
            }
            return group -> default_debug;
         }
      }
      it1 ++;
   }
   else // search in stand alone functions
   {
      list< tnlDebugEntry* > :: iterator it = alone_entries. begin();
      while( it != alone_entries. end() )
      {
         tnlDebugEntry* entry = * it;
         if( entry -> function_name == function_name )
            return entry -> debug;
         it ++;
      }
   }
   return default_debug;
}
//--------------------------------------------------------------------------
bool tnlDebugStructure :: Interactive( const char* group_name,
                                       const char* function_name )
{
   if( ! interactive ) return false;

   if( strlen( group_name ) != 0 )
   {
      list< tnlDebugGroup* > :: iterator it1 = debug_groups. begin();
      while( it1 != debug_groups. end() )
      {
         tnlDebugGroup* group = * it1;
         if( group -> group_name == group_name )
         {
            if( ! group -> debug ) return false;
            list< tnlDebugEntry* > :: iterator it2 = group -> debug_entries. begin();
            while( it2 != group -> debug_entries. end() )
            {
               tnlDebugEntry* entry = * it2;
               if( entry -> function_name == function_name )
                  return entry -> interactive;
               it2 ++;
            }
            return group -> default_interactive;
         }
         it1 ++;
      }
   }
   else // search in stand alone functions
   {
      list< tnlDebugEntry* > :: iterator it = alone_entries. begin();
      while( it != alone_entries. end() )
      {
         tnlDebugEntry* entry = * it;
         if( entry -> function_name == function_name )
            return entry -> interactive;
         it ++;
      }
   }
   return default_interactive;
}
//--------------------------------------------------------------------------
void tnlDebugStructure :: Print()
{
   list< tnlDebugGroup* > :: iterator it1 = debug_groups. begin();
   while( it1 != debug_groups. end() )
   {
      tnlDebugGroup* grp = * it1;
      cout << "class " << grp -> group_name
           << "[ debug = " <<  booltostr( grp -> debug )
           << ", default = " << booltostr( grp -> default_debug )
           << " ]" << endl << "{" << endl;
      list< tnlDebugEntry* > :: iterator it2 = grp -> debug_entries. begin();
      while( it2 != grp -> debug_entries. end() )
      {
         cout << "   " << ( * it2 ) -> function_name
              << "[ debug = " << booltostr( ( * it2 ) -> debug )
              << " ]" << endl;  
         it2 ++;
      }
      cout << "}" << endl;
      it1 ++;
   }
}
