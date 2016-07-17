/***************************************************************************
                          tnlConfigDescription.cpp  -  description
                             -------------------
    begin                : 2007/06/09
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <fstream>
#include <iomanip>
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>
#include <core/mfuncs.h>

namespace TNL {

tnlConfigDescription :: tnlConfigDescription()
: currentEntry( 0 )
{
}

tnlConfigDescription :: ~tnlConfigDescription()
{
   entries. DeepEraseAll();
}

void tnlConfigDescription::printUsage( const char* program_name ) const
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

void tnlConfigDescription :: addMissingEntries( tnlParameterContainer& parameter_container ) const
{
   int i;
   const int size = entries.getSize();
   for( i = 0; i < size; i ++ )
   {
      const char* entry_name = entries[ i ]->name.getString();
      if( entries[ i ]->hasDefaultValue &&
          ! parameter_container.checkParameter( entry_name ) )
      {
         if( entries[ i ]->getEntryType() == "tnlString" )
         {
            parameter_container. addParameter< tnlString >(
               entry_name,
               ( ( tnlConfigEntry< tnlString >* ) entries[ i ] ) -> defaultValue );
            continue;
         }
         if( entries[ i ]->getEntryType() == "bool" )
         {
            parameter_container. addParameter< bool >(
               entry_name,
               ( ( tnlConfigEntry< bool >* ) entries[ i ] ) -> defaultValue );
            continue;
         }
         if( entries[ i ]->getEntryType() == "int" )
         {
            parameter_container. addParameter< int >(
               entry_name,
               ( ( tnlConfigEntry< int >* ) entries[ i ] ) -> defaultValue );
            continue;
         }
         if( entries[ i ]->getEntryType() == "double" )
         {
            parameter_container. addParameter< double >(
               entry_name,
               ( ( tnlConfigEntry< double >* ) entries[ i ] ) -> defaultValue );
            continue;
         }
      }
   }
}

bool tnlConfigDescription :: checkMissingEntries( tnlParameterContainer& parameter_container,
                                                  bool printUsage,
                                                  const char* programName ) const
{
   int i;
   const int size = entries. getSize();
   tnlList< tnlString > missingParameters;
   for( i = 0; i < size; i ++ )
   {
      const char* entry_name = entries[ i ] -> name. getString();
      if( entries[ i ] -> required &&
          ! parameter_container. checkParameter( entry_name ) )
         missingParameters.Append( entry_name );
   }
   if( missingParameters.getSize() != 0 )
   {
      cerr << "Some mandatory parameters are misssing. They are listed at the end. " << endl;
      if( printUsage )
         this->printUsage( programName );
      cerr << "Add the following missing  parameters to the command line: " << endl << "   ";
      for( int i = 0; i < missingParameters.getSize(); i++ )
         cerr << "--" << missingParameters[ i ] << " ... ";
      cerr << endl;
      return false;
   }
   return true;
}

} // namespace TNL

