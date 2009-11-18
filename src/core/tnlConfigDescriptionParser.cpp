/***************************************************************************
                          tnlConfigDescriptionParser.cpp  -  description
                             -------------------
    begin                : 2007/06/13
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

#include "tnlConfigDescriptionParser.h"
#include "tnlConfigDescriptionScanner.h"
#include "debug.h"
#include <assert.h>

tnlConfigDescriptionParser* tnlConfigDescriptionParser :: current_parser;

//--------------------------------------------------------------------------
tnlConfigDescriptionParser :: tnlConfigDescriptionParser()
{
   scanner = new mCDSFlexLexer();
}
//--------------------------------------------------------------------------
tnlConfigDescriptionParser :: ~tnlConfigDescriptionParser()
{
   assert( scanner );
   delete scanner;
}
//--------------------------------------------------------------------------
void tnlConfigDescriptionParser :: setScanner( istream* in_stream )
{
   scanner -> switch_streams( in_stream, 0 );
}
//--------------------------------------------------------------------------
int tnlConfigDescriptionParser :: runParsing( tnlConfigDescription* conf_desc )
{
   config_description = conf_desc;
   current_parser = this;
   line = 1;
   current_entry_type. Reset();
   return parse();
}
//--------------------------------------------------------------------------
int tnlConfigDescriptionParser :: lex()
{
   return scanner -> yylex();
}
//--------------------------------------------------------------------------
void tnlConfigDescriptionParser :: newLine()
{
   line ++;
   //cout << "New line " << line << endl;
}
//--------------------------------------------------------------------------
void tnlConfigDescriptionParser :: setSVal( char* s )
{
   d_val__. s_val = s;
   //cout << " d_val. s_val is " << d_val. s_val << endl;
}
//--------------------------------------------------------------------------
void tnlConfigDescriptionParser :: setIVal( char* s )
{
   d_val__. i_val = atoi( s );
   //cout << " d_val. i_val is " << d_val. i_val << endl;
}
//--------------------------------------------------------------------------
void tnlConfigDescriptionParser :: setDVal( char* s )
{
   d_val__. d_val = atof( s );
   //cout << " d_val. d_val is " << d_val. d_val << endl;
}
//--------------------------------------------------------------------------
void tnlConfigDescriptionParser :: setBVal( bool b  )
{
   d_val__. b_val = b;
   //cout << " d_val. d_val is " << d_val. d_val << endl;
}
//--------------------------------------------------------------------------
void tnlConfigDescriptionParser :: SetCurrentGroupId( const char* id )
{
    current_group_name. SetString( id );
}
//--------------------------------------------------------------------------
void tnlConfigDescriptionParser :: SetCurrentGroupComment( const char* comment )
{
    current_group_comment. SetString( comment,
                                      1, // prefix cut off for '['
                                      1  // sufix cut off for ']'
                                      );
}
//--------------------------------------------------------------------------
void tnlConfigDescriptionParser :: AddCurrentGroup()
{
   //cout << "Adding group with ID " << current_group_name << 
   //        " and comment " << current_group_comment << endl;
   config_description -> AddGroup( current_group_name. Data(),
                                   current_group_comment. Data() );
   current_group_name. SetString( 0 );
   current_group_comment. SetString( 0 );
}
//--------------------------------------------------------------------------
void tnlConfigDescriptionParser :: SetCurrentEntryTypeName( const char* _basic_type )
{
   current_entry_type. basic_type. SetString( _basic_type );
}
//--------------------------------------------------------------------------
void tnlConfigDescriptionParser :: SetCurrentEntryTypeIsList( const bool _list_entry )
{
   current_entry_type. list_entry = _list_entry; 
}
//--------------------------------------------------------------------------
void tnlConfigDescriptionParser :: SetCurrentEntryId( const char* id )
{
   current_entry_name. SetString( id );
}
//--------------------------------------------------------------------------
void tnlConfigDescriptionParser :: SetCurrentEntryComment( const char* comment )
{
   current_entry_comment. SetString( comment,
                                     1, // prefix cut off for '['
                                     1  // sufix cut off for ']'
                                     );
}
//--------------------------------------------------------------------------
void tnlConfigDescriptionParser :: AddCurrentEntry( bool required )
{
   config_description -> AddEntry( current_entry_name. Data(),
                                   current_entry_type,
                                   current_group_name. Data(),
                                   current_entry_comment. Data(),
                                   required );
   current_entry_type. Reset();
}
//--------------------------------------------------------------------------
void tnlConfigDescriptionParser :: AddCurrentEntryWithDefaultValue()
{
   if( current_entry_type. basic_type == "string" )
   {
      config_description -> AddEntryWithDefaultValue(
                                   current_entry_name. Data(),
                                   current_entry_type,
                                   current_group_name. Data(),
                                   current_entry_comment. Data(),
                                   string_default_value );
      current_entry_type. Reset();
      return;
   }
   if( current_entry_type. basic_type == "integer" )
   {
      config_description -> AddEntryWithDefaultValue(
                                   current_entry_name. Data(),
                                   current_entry_type,
                                   current_group_name. Data(),
                                   current_entry_comment. Data(),
                                   integer_default_value );
      current_entry_type. Reset();
      return;
   }
   if( current_entry_type. basic_type == "real" )
   {
      config_description -> AddEntryWithDefaultValue(
                                   current_entry_name. Data(),
                                   current_entry_type,
                                   current_group_name. Data(),
                                   current_entry_comment. Data(),
                                   real_default_value );
      current_entry_type. Reset();
      return;
   }
   if( current_entry_type. basic_type == "bool" )
   {
      config_description -> AddEntryWithDefaultValue(
                                   current_entry_name. Data(),
                                   current_entry_type,
                                   current_group_name. Data(),
                                   current_entry_comment. Data(),
                                   bool_default_value );
      current_entry_type. Reset();
   }
}
//--------------------------------------------------------------------------
