/***************************************************************************
                          tnlDebugParser.cpp  -  description
                             -------------------
    begin                : 2009/08/11
    copyright            : (C) 2009 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlDebugParser.h>
#include <TNL/tnlDebugScanner.h>
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
   //cout << "New line " << line << std::endl;
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
   //cout << " d_val. d_val is " << d_val. d_val << std::endl;
}
//--------------------------------------------------------------------------
void tnlDebugParser :: AddCurrentGroup()
{
#ifdef DEBUG
   std::cout << "Inserting class " << current_group -> group_name << std::endl;
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
   std::cout << "Inserting function " << current_entry -> function_name;
#endif
   if( current_group )
   {
#ifdef DEBUG
      std::cout << " into class " << current_group -> group_name << std::endl;
#endif
      current_group -> debug_entries. push_back( current_entry );
   }
   else
   {
#ifdef DEBUG
      std::cout << " as stand-alone function " << std::endl;
#endif
      debug_structure -> AppendAloneEntry( current_entry );
   }
   current_entry = 0;
}
//--------------------------------------------------------------------------
void tnlDebugParser :: SetCurrentClassId( char* s )
{
#ifdef DEBUG
   std::cout << "Setting current class id to " << s << std::endl;
#endif
   current_group = new tnlDebugGroup;
   current_group -> group_name = string( s );
}
//--------------------------------------------------------------------------
void tnlDebugParser :: SetCurrentFunctionId( char* s )
{
#ifdef DEBUG
   std::cout << "Setting current function id to " << s << std::endl;
#endif
   assert( current_group );
   current_entry = new tnlDebugEntry;
   current_entry -> function_name = string( s );
}
//--------------------------------------------------------------------------
void tnlDebugParser :: SetBool( bool v )
{
#ifdef DEBUG
   std::cout << "Setting bool = " << v << std::endl;
#endif
   bool_value = v;
}
//--------------------------------------------------------------------------
void tnlDebugParser :: SetDebugValue( )
{
#ifdef DEBUG
   std::cout << "Setting debug value = " << bool_value << std::endl;
#endif
   debug_value = bool_value;
}
//--------------------------------------------------------------------------
void tnlDebugParser :: SetDefaultDebugValue( )
{
#ifdef DEBUG
   std::cout << "Setting default debug value = " << bool_value << std::endl;
#endif
   default_debug_value = bool_value;
}
//--------------------------------------------------------------------------
void tnlDebugParser :: SetClassDebugSettings()
{
   assert( current_group );
#ifdef DEBUG
   std::cout << "Setting class debug value = " << debug_value << std::endl;
   std::cout << "Setting class default debug value = " << default_debug_value << std::endl;
#endif
   current_group -> debug = debug_value;
   current_group -> default_debug = default_debug_value;

   debug_value = false;
   default_debug_value = false;
}
