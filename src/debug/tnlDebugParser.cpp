/***************************************************************************
                          tnlDebugParser.cpp  -  description
                             -------------------
    begin                : 2009/08/11
    copyright            : (C) 2009 by Tomá¹ Oberhuber
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

#include "tnlDebugParser.h"
#include "tnlDebugScanner.h"
#include "assert.h"

tnlDebugParser* tnlDebugParser :: current_parser;
//--------------------------------------------------------------------------
tnlDebugParser :: tnlDebugParser()
   : bool_value( false ),
     debug_value( false ),
     default_debug_value( false ),
     current_group( 0 ),
     current_entry( 0 )
{
   scanner = new tnlDebugFlexLexer();
}
//--------------------------------------------------------------------------
tnlDebugParser :: ~tnlDebugParser()
{
   assert( scanner );
   delete scanner;
}
//--------------------------------------------------------------------------
void tnlDebugParser :: setScanner( istream* in_stream )
{
   scanner -> switch_streams( in_stream, 0 );
}
//--------------------------------------------------------------------------
int tnlDebugParser :: runParsing( tnlDebugStructure* _debug_structure )
{
   debug_structure = _debug_structure;
   current_parser = this;
   line = 1;
   //current_entry_type. Reset();
   return parse();
}
//--------------------------------------------------------------------------
int tnlDebugParser :: lex()
{
   return scanner -> yylex();
}
//--------------------------------------------------------------------------
void tnlDebugParser :: newLine()
{
   line ++;
   //cout << "New line " << line << endl;
}
//--------------------------------------------------------------------------
void tnlDebugParser :: setSVal( char* s )
{
   d_val__. s_val = s;
}
//--------------------------------------------------------------------------
void tnlDebugParser :: setBVal( bool b  )
{
   d_val__. b_val = b;
   //cout << " d_val. d_val is " << d_val. d_val << endl;
}
//--------------------------------------------------------------------------
void tnlDebugParser :: AddCurrentGroup()
{
#ifdef DEBUG
   cout << "Inserting class " << current_group -> group_name << endl;
#endif
   debug_structure -> AppendGroup( current_group );
   current_group = 0;
   
}
//--------------------------------------------------------------------------
void tnlDebugParser :: AddCurrentEntry()
{
   assert( current_entry );
   current_entry -> debug = debug_value;
   debug_value = false;
#ifdef DEBUG
   cout << "Inserting function " << current_entry -> function_name;
#endif
   if( current_group )
   {
#ifdef DEBUG
      cout << " into class " << current_group -> group_name << endl;
#endif
      current_group -> debug_entries. push_back( current_entry ); 
   }
   else
   {
#ifdef DEBUG
      cout << " as stand-alone function " << endl;
#endif
      debug_structure -> AppendAloneEntry( current_entry );
   }
   current_entry = 0;
}
//--------------------------------------------------------------------------
void tnlDebugParser :: SetCurrentClassId( char* s )
{
#ifdef DEBUG
   cout << "Setting current class id to " << s << endl;
#endif
   current_group = new tnlDebugGroup;
   current_group -> group_name = string( s );
}
//--------------------------------------------------------------------------
void tnlDebugParser :: SetCurrentFunctionId( char* s )
{
#ifdef DEBUG
   cout << "Setting current function id to " << s << endl;
#endif
   assert( current_group );
   current_entry = new tnlDebugEntry;
   current_entry -> function_name = string( s );   
}
//--------------------------------------------------------------------------
void tnlDebugParser :: SetBool( bool v )
{
#ifdef DEBUG
   cout << "Setting bool = " << v << endl;
#endif
   bool_value = v;
}
//--------------------------------------------------------------------------
void tnlDebugParser :: SetDebugValue( )
{
#ifdef DEBUG
   cout << "Setting debug value = " << bool_value << endl;
#endif
   debug_value = bool_value;
}
//--------------------------------------------------------------------------
void tnlDebugParser :: SetDefaultDebugValue( )
{
#ifdef DEBUG
   cout << "Setting default debug value = " << bool_value << endl;
#endif
   default_debug_value = bool_value;
}
//--------------------------------------------------------------------------
void tnlDebugParser :: SetClassDebugSettings()
{
   assert( current_group );
#ifdef DEBUG
   cout << "Setting class debug value = " << debug_value << endl;
   cout << "Setting class default debug value = " << default_debug_value << endl;
#endif
   current_group -> debug = debug_value;
   current_group -> default_debug = default_debug_value;

   debug_value = false;
   default_debug_value = false;
}
