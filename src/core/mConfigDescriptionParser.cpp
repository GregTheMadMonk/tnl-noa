/***************************************************************************
                          mConfigDescriptionParser.cpp  -  description
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

#include "mConfigDescriptionParser.h"
#include "mConfigDescriptionScanner.h"
#include "debug.h"
#include <assert.h>

mConfigDescriptionParser* mConfigDescriptionParser :: current_parser;

//--------------------------------------------------------------------------
mConfigDescriptionParser :: mConfigDescriptionParser()
{
   scanner = new mCDSFlexLexer();
}
//--------------------------------------------------------------------------
mConfigDescriptionParser :: ~mConfigDescriptionParser()
{
   assert( scanner );
   delete scanner;
}
//--------------------------------------------------------------------------
void mConfigDescriptionParser :: setScanner( istream* in_stream )
{
   scanner -> switch_streams( in_stream, 0 );
}
//--------------------------------------------------------------------------
int mConfigDescriptionParser :: runParsing( mConfigDescription* conf_desc )
{
   config_description = conf_desc;
   current_parser = this;
   line = 1;
   current_entry_type. Reset();
   return parse();
}
//--------------------------------------------------------------------------
int mConfigDescriptionParser :: lex()
{
   return scanner -> yylex();
}
//--------------------------------------------------------------------------
void mConfigDescriptionParser :: newLine()
{
   line ++;
   //cout << "New line " << line << endl;
}
//--------------------------------------------------------------------------
void mConfigDescriptionParser :: setSVal( char* s )
{
   d_val__. s_val = s;
   //cout << " d_val. s_val is " << d_val. s_val << endl;
}
//--------------------------------------------------------------------------
void mConfigDescriptionParser :: setIVal( char* s )
{
   d_val__. i_val = atoi( s );
   //cout << " d_val. i_val is " << d_val. i_val << endl;
}
//--------------------------------------------------------------------------
void mConfigDescriptionParser :: setDVal( char* s )
{
   d_val__. d_val = atof( s );
   //cout << " d_val. d_val is " << d_val. d_val << endl;
}
//--------------------------------------------------------------------------
void mConfigDescriptionParser :: setBVal( bool b  )
{
   d_val__. b_val = b;
   //cout << " d_val. d_val is " << d_val. d_val << endl;
}
//--------------------------------------------------------------------------
void mConfigDescriptionParser :: SetCurrentGroupId( const char* id )
{
    current_group_name. SetString( id );
}
//--------------------------------------------------------------------------
void mConfigDescriptionParser :: SetCurrentGroupComment( const char* comment )
{
    current_group_comment. SetString( comment,
                                      1, // prefix cut off for '['
                                      1  // sufix cut off for ']'
                                      );
}
//--------------------------------------------------------------------------
void mConfigDescriptionParser :: AddCurrentGroup()
{
   //cout << "Adding group with ID " << current_group_name << 
   //        " and comment " << current_group_comment << endl;
   config_description -> AddGroup( current_group_name. Data(),
                                   current_group_comment. Data() );
   current_group_name. SetString( 0 );
   current_group_comment. SetString( 0 );
}
//--------------------------------------------------------------------------
void mConfigDescriptionParser :: SetCurrentEntryTypeName( const char* _basic_type )
{
   current_entry_type. basic_type. SetString( _basic_type );
}
//--------------------------------------------------------------------------
void mConfigDescriptionParser :: SetCurrentEntryTypeIsList( const bool _list_entry )
{
   current_entry_type. list_entry = _list_entry; 
}
//--------------------------------------------------------------------------
void mConfigDescriptionParser :: SetCurrentEntryId( const char* id )
{
   current_entry_name. SetString( id );
}
//--------------------------------------------------------------------------
void mConfigDescriptionParser :: SetCurrentEntryComment( const char* comment )
{
   current_entry_comment. SetString( comment,
                                     1, // prefix cut off for '['
                                     1  // sufix cut off for ']'
                                     );
}
//--------------------------------------------------------------------------
void mConfigDescriptionParser :: AddCurrentEntry( bool required )
{
   config_description -> AddEntry( current_entry_name. Data(),
                                   current_entry_type,
                                   current_group_name. Data(),
                                   current_entry_comment. Data(),
                                   required );
   current_entry_type. Reset();
}
//--------------------------------------------------------------------------
void mConfigDescriptionParser :: AddCurrentEntryWithDefaultValue()
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
