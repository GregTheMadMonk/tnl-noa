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
#include <config/tnlConfigDescriptionParser.h>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>
#include <core/mfuncs.h>


tnlConfigDescription :: tnlConfigDescription()
: currentEntry( 0 )
{
}

tnlConfigDescription :: ~tnlConfigDescription()
{
   entries. DeepEraseAll();
}

/*const tnlString tnlConfigDescription :: getEntryType( const char* name ) const
{
   int i;
   const int size = entries.getSize();
   for( i = 0; i < size; i ++ )
      if( entries[ i ]->name == name )
         return entries[ i ]->getEntryType();
   return tnlString( "" );
}*/

void tnlConfigDescription::printUsage( const char* program_name )
{
   cout << "Usage of: " << program_name << endl << endl;
   int i, j;
   //const int group_num = groups. getSize();
   const int entries_num = entries. getSize();
   int max_name_length( 0 );
   int max_type_length( 0 );
   for( j = 0; j < entries_num; j ++ )
      if( ! entries[ j ]->isDelimiter() )
      {
         max_name_length = Max( max_name_length,
                     entries[ j ] -> name. getLength() );
         max_type_length = Max( max_type_length,
                     entries[ j ] -> getUIEntryType().getLength() );
      }
   max_name_length += 2; // this is for '--'

   for( j = 0; j < entries_num; j ++ )
   {
      if( entries[ j ]->isDelimiter() )
      {
         cout << endl;
         cout << entries[ j ]->description;
         cout << endl << endl;
      }
      else
      {
         cout << setw( max_name_length + 3 ) << tnlString( "--" ) + entries[ j ]->name
              << setw( max_type_length + 5 ) << entries[ j ] -> getUIEntryType()
              << "    " << entries[ j ]->description;
         if( entries[ j ] -> required )
            cout << " *** REQUIRED ***";
         if( entries[ j ]->hasEnumValues() )
         {
            cout << endl
                 << setw( max_name_length + 3 ) << ""
                 << setw( max_type_length + 5 ) << ""
                 << "    ";
            entries[ j ]->printEnumValues();
         }
         if( entries[ j ]->hasDefaultValue )
         {
            cout << endl
                 << setw( max_name_length + 3 ) << ""
                 << setw( max_type_length + 5 ) << ""
                 << "    ";
            cout << "- Default value is: " << entries[ j ]->printDefaultValue();
         }
         cout << endl;
      }
   }
   cout << endl;
}
//--------------------------------------------------------------------------
bool tnlConfigDescription::parseConfigDescription( const char* file_name )
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
void tnlConfigDescription :: addMissingEntries( tnlParameterContainer& parameter_container ) const
{
   int i;
   const int size = entries.getSize();
   for( i = 0; i < size; i ++ )
   {
      const char* entry_name = entries[ i ]->name.getString();
      if( entries[ i ]->hasDefaultValue &&
          ! parameter_container.CheckParameter( entry_name ) )
      {
         if( entries[ i ]->getEntryType() == "tnlString" )
         {
            parameter_container. AddParameter< tnlString >(
               entry_name,
               ( ( tnlConfigEntry< tnlString >* ) entries[ i ] ) -> defaultValue );
            continue;
         }
         if( entries[ i ]->getEntryType() == "bool" )
         {
            parameter_container. AddParameter< bool >(
               entry_name,
               ( ( tnlConfigEntry< bool >* ) entries[ i ] ) -> defaultValue );
            continue;
         }
         if( entries[ i ]->getEntryType() == "int" )
         {
            parameter_container. AddParameter< int >(
               entry_name,
               ( ( tnlConfigEntry< int >* ) entries[ i ] ) -> defaultValue );
            continue;
         }
         if( entries[ i ]->getEntryType() == "double" )
         {
            parameter_container. AddParameter< double >(
               entry_name,
               ( ( tnlConfigEntry< double >* ) entries[ i ] ) -> defaultValue );
            continue;
         }
      }
   }
}
//--------------------------------------------------------------------------
bool tnlConfigDescription :: checkMissingEntries( tnlParameterContainer& parameter_container ) const
{
   int i;
   const int size = entries. getSize();
   bool missing_parameter( false );
   for( i = 0; i < size; i ++ )
   {
      const char* entry_name = entries[ i ] -> name. getString();
      if( entries[ i ] -> required && 
          ! parameter_container. CheckParameter( entry_name ) )
      {
         cerr << "Missing parameter " << entry_name << "." << endl;
         missing_parameter = true;
      }
   }
   return ! missing_parameter;
}

