/***************************************************************************
                          mConfigDescription.cpp  -  description
                             -------------------
    begin                : 2007/06/09
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

#include <fstream>
#include <iomanip>
#include "mConfigDescriptionParser.h"
#include "mConfigDescription.h"
#include "mParameterContainer.h"
#include "mfuncs.h"

//--------------------------------------------------------------------------
mConfigDescription :: mConfigDescription()
{
}
//--------------------------------------------------------------------------
mConfigDescription :: ~mConfigDescription()
{
   groups. DeepEraseAll();
   entries. DeepEraseAll();
}
//--------------------------------------------------------------------------
void mConfigDescription :: AddGroup( const char* name,
                                     const char* description )
{
   groups. Append( new mConfigGroup( name, description ) );
}
//--------------------------------------------------------------------------
void mConfigDescription :: AddEntry( const char* name,
                                     const mConfigEntryType& type,
                                     const char* group,
                                     const char* comment,
                                     bool required )
{
   entries. Append( new mConfigEntryBase( name, type, group, comment, required ) );
}
//--------------------------------------------------------------------------
const mConfigEntryType* mConfigDescription :: GetEntryType( const char* name ) const
{
   int i;
   const int size = entries. Size();
   for( i = 0; i < size; i ++ )
      if( entries[ i ] -> name == name )
         return &( entries[ i ] -> type );
   return NULL;
}
//--------------------------------------------------------------------------
void mConfigDescription :: PrintUsage( const char* program_name )
{
   cout << "Usage of: " << program_name << endl << endl;
   int i, j;
   const int group_num = groups. Size();
   const int entries_num = entries. Size();
   for( i = 0; i < group_num; i ++ )
   {
      const char* group_name = groups[ i ] -> name. Data();
      cout << groups[ i ] -> comment << endl;
      int max_name_length( 0 );
      int max_type_length( 0 );
      for( j = 0; j < entries_num; j ++ )
         if( entries[ j ] -> group == group_name )
         {
            max_name_length = Max( max_name_length, 
                        entries[ j ] -> name. Length() );
            max_type_length = Max( max_type_length, 
                        entries[ j ] -> type. basic_type. Length() );
         }
            
      for( j = 0; j < entries_num; j ++ )
      {
         if( entries[ j ] -> group == group_name )
         {
            cout << setw( max_name_length + 3 ) << entries[ j ] -> name 
                 << setw( max_type_length + 5 ) << entries[ j ] -> type. basic_type   
                 << "    " << entries[ j ] -> comment;
            if( entries[ j ] -> has_default_value )
            {
               cout << " DEFAULT VALUE IS: ";
               if( entries[ j ] -> type. basic_type == "string" )
                  cout << ( ( mConfigEntry< mString >* ) entries[ j ] ) -> default_value;
               if( entries[ j ] -> type. basic_type == "integer" )
                  cout << ( ( mConfigEntry< int >* ) entries[ j ] ) -> default_value;
               if( entries[ j ] -> type. basic_type == "real" )
                  cout << ( ( mConfigEntry< double >* ) entries[ j ] ) -> default_value;
               if( entries[ j ] -> type. basic_type == "bool" )
                  if( ( ( mConfigEntry< bool >* ) entries[ j ] ) -> default_value )
                     cout << "yes";
                  else cout << "no";
            }
            if( entries[ j ] -> required )
               cout << " REQUIRED."; 
            cout << endl;
         }
      }
      cout << endl;
   }
}
//--------------------------------------------------------------------------
bool mConfigDescription :: ParseConfigDescription( const char* file_name )
{
   mConfigDescriptionParser parser;
   fstream in_file;
   in_file. open( file_name, ios :: in );
   if( ! in_file )
   {
      cerr << "Unable to open the file " << file_name << endl;
      return false;
   }
   parser. setScanner( &in_file );
   if( ! parser. runParsing( this ) ) return false;
   return true;
}
//--------------------------------------------------------------------------
void mConfigDescription :: AddMissingEntries( mParameterContainer& parameter_container ) const
{
   int i;
   const int size = entries. Size();
   for( i = 0; i < size; i ++ )
   {
      const char* entry_name = entries[ i ] -> name. Data();
      if( entries[ i ] -> has_default_value && 
          ! parameter_container. CheckParameter( entry_name ) )
      {
         if( entries[ i ] -> type. basic_type == "string" )
         {
            parameter_container. AddParameter< mString >(
               entry_name,
               ( ( mConfigEntry< mString >* ) entries[ i ] ) -> default_value );
            continue;
         }
         if( entries[ i ] -> type. basic_type == "bool" )
         {
            parameter_container. AddParameter< bool >(
               entry_name,
               ( ( mConfigEntry< bool >* ) entries[ i ] ) -> default_value );
            continue;
         }
         if( entries[ i ] -> type. basic_type == "integer" )
         {
            parameter_container. AddParameter< int >(
               entry_name,
               ( ( mConfigEntry< int >* ) entries[ i ] ) -> default_value );
            continue;
         }
         if( entries[ i ] -> type. basic_type == "real" )
         {
            parameter_container. AddParameter< double >(
               entry_name,
               ( ( mConfigEntry< double >* ) entries[ i ] ) -> default_value );
            continue;
         }
      }
   }
}
//--------------------------------------------------------------------------
bool mConfigDescription :: CheckMissingEntries( mParameterContainer& parameter_container ) const
{
   int i;
   const int size = entries. Size();
   bool missing_parameter( false );
   for( i = 0; i < size; i ++ )
   {
      const char* entry_name = entries[ i ] -> name. Data();
      if( entries[ i ] -> required && 
          ! parameter_container. CheckParameter( entry_name ) )
      {
         cerr << "Missing parameter " << entry_name << "." << endl;
         missing_parameter = true;
      }
   }
   return ! missing_parameter;
}

