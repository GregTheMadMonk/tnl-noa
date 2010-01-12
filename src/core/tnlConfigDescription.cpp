/***************************************************************************
                          tnlConfigDescription.cpp  -  description
                             -------------------
    begin                : 2007/06/09
    copyright            : (C) 2007 by Tomas Oberhuber
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

#include <fstream>
#include <iomanip>
#include "tnlConfigDescriptionParser.h"
#include "tnlConfigDescription.h"
#include "tnlParameterContainer.h"
#include "mfuncs.h"

//--------------------------------------------------------------------------
tnlConfigDescription :: tnlConfigDescription()
{
}
//--------------------------------------------------------------------------
tnlConfigDescription :: ~tnlConfigDescription()
{
   groups. DeepEraseAll();
   entries. DeepEraseAll();
}
//--------------------------------------------------------------------------
void tnlConfigDescription :: AddGroup( const char* name,
                                     const char* description )
{
   groups. Append( new tnlConfigGroup( name, description ) );
}
//--------------------------------------------------------------------------
void tnlConfigDescription :: AddEntry( const char* name,
                                     const tnlConfigEntryType& type,
                                     const char* group,
                                     const char* comment,
                                     bool required )
{
   entries. Append( new tnlConfigEntryBase( name, type, group, comment, required ) );
}
//--------------------------------------------------------------------------
const tnlConfigEntryType* tnlConfigDescription :: GetEntryType( const char* name ) const
{
   int i;
   const int size = entries. Size();
   for( i = 0; i < size; i ++ )
      if( entries[ i ] -> name == name )
         return &( entries[ i ] -> type );
   return NULL;
}
//--------------------------------------------------------------------------
void tnlConfigDescription :: PrintUsage( const char* program_name )
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
                  cout << ( ( tnlConfigEntry< tnlString >* ) entries[ j ] ) -> default_value;
               if( entries[ j ] -> type. basic_type == "integer" )
                  cout << ( ( tnlConfigEntry< int >* ) entries[ j ] ) -> default_value;
               if( entries[ j ] -> type. basic_type == "real" )
                  cout << ( ( tnlConfigEntry< double >* ) entries[ j ] ) -> default_value;
               if( entries[ j ] -> type. basic_type == "bool" )
                  if( ( ( tnlConfigEntry< bool >* ) entries[ j ] ) -> default_value )
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
bool tnlConfigDescription :: ParseConfigDescription( const char* file_name )
{
   tnlConfigDescriptionParser parser;
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
void tnlConfigDescription :: AddMissingEntries( tnlParameterContainer& parameter_container ) const
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
            parameter_container. AddParameter< tnlString >(
               entry_name,
               ( ( tnlConfigEntry< tnlString >* ) entries[ i ] ) -> default_value );
            continue;
         }
         if( entries[ i ] -> type. basic_type == "bool" )
         {
            parameter_container. AddParameter< bool >(
               entry_name,
               ( ( tnlConfigEntry< bool >* ) entries[ i ] ) -> default_value );
            continue;
         }
         if( entries[ i ] -> type. basic_type == "integer" )
         {
            parameter_container. AddParameter< int >(
               entry_name,
               ( ( tnlConfigEntry< int >* ) entries[ i ] ) -> default_value );
            continue;
         }
         if( entries[ i ] -> type. basic_type == "real" )
         {
            parameter_container. AddParameter< double >(
               entry_name,
               ( ( tnlConfigEntry< double >* ) entries[ i ] ) -> default_value );
            continue;
         }
      }
   }
}
//--------------------------------------------------------------------------
bool tnlConfigDescription :: CheckMissingEntries( tnlParameterContainer& parameter_container ) const
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

