/***************************************************************************
                          Config::ConfigDescription.cpp  -  description
                             -------------------
    begin                : 2007/06/09
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <fstream>
#include <iomanip>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>

namespace TNL {
namespace Config {    

ConfigDescription :: ConfigDescription()
: currentEntry( 0 )
{
}

ConfigDescription :: ~ConfigDescription()
{
   entries.DeepEraseAll();
}

void ConfigDescription::printUsage( const char* program_name ) const
{
   std::cout << "Usage of: " << program_name << std::endl << std::endl;
   int i, j;
   //const int group_num = groups. getSize();
   const int entries_num = entries. getSize();
   int max_name_length( 0 );
   int max_type_length( 0 );
   for( j = 0; j < entries_num; j ++ )
      if( ! entries[ j ]->isDelimiter() )
      {
         max_name_length = std::max( max_name_length,
                     entries[ j ] -> name. getLength() );
         max_type_length = std::max( max_type_length,
                     entries[ j ] -> getUIEntryType().getLength() );
      }
   max_name_length += 2; // this is for '--'

   for( j = 0; j < entries_num; j ++ )
   {
      if( entries[ j ]->isDelimiter() )
      {
         std::cout << std::endl;
         std::cout << entries[ j ]->description;
         std::cout << std::endl << std::endl;
      }
      else
      {
         std::cout << std::setw( max_name_length + 3 ) << String( "--" ) + entries[ j ]->name
              << std::setw( max_type_length + 5 ) << entries[ j ] -> getUIEntryType()
              << "    " << entries[ j ]->description;
         if( entries[ j ] -> required )
            std::cout << " *** REQUIRED ***";
         if( entries[ j ]->hasEnumValues() )
         {
            std::cout << std::endl
                 << std::setw( max_name_length + 3 ) << ""
                 << std::setw( max_type_length + 5 ) << ""
                 << "    ";
            entries[ j ]->printEnumValues();
         }
         if( entries[ j ]->hasDefaultValue )
         {
            std::cout << std::endl
                 << std::setw( max_name_length + 3 ) << ""
                 << std::setw( max_type_length + 5 ) << ""
                 << "    ";
            std::cout << "- Default value is: " << entries[ j ]->printDefaultValue();
         }
         std::cout << std::endl;
      }
   }
   std::cout << std::endl;
}

void Config::ConfigDescription :: addMissingEntries( Config::ParameterContainer& parameter_container ) const
{
   int i;
   const int size = entries.getSize();
   for( i = 0; i < size; i ++ )
   {
      const char* entry_name = entries[ i ]->name.getString();
      if( entries[ i ]->hasDefaultValue &&
          ! parameter_container.checkParameter( entry_name ) )
      {
         if( entries[ i ]->getEntryType() == "String" )
         {
            parameter_container. addParameter< String >(
               entry_name,
               ( ( ConfigEntry< String >* ) entries[ i ] ) -> defaultValue );
            continue;
         }
         if( entries[ i ]->getEntryType() == "bool" )
         {
            parameter_container. addParameter< bool >(
               entry_name,
               ( ( ConfigEntry< bool >* ) entries[ i ] ) -> defaultValue );
            continue;
         }
         if( entries[ i ]->getEntryType() == "int" )
         {
            parameter_container. addParameter< int >(
               entry_name,
               ( ( ConfigEntry< int >* ) entries[ i ] ) -> defaultValue );
            continue;
         }
         if( entries[ i ]->getEntryType() == "double" )
         {
            parameter_container. addParameter< double >(
               entry_name,
               ( ( ConfigEntry< double >* ) entries[ i ] ) -> defaultValue );
            continue;
         }
      }
   }
}

bool Config::ConfigDescription :: checkMissingEntries( Config::ParameterContainer& parameter_container,
                                                  bool printUsage,
                                                  const char* programName ) const
{
   int i;
   const int size = entries. getSize();
   List< String > missingParameters;
   for( i = 0; i < size; i ++ )
   {
      const char* entry_name = entries[ i ] -> name. getString();
      if( entries[ i ] -> required &&
          ! parameter_container. checkParameter( entry_name ) )
         missingParameters.Append( entry_name );
   }
   if( missingParameters.getSize() != 0 )
   {
      std::cerr << "Some mandatory parameters are misssing. They are listed at the end. " << std::endl;
      if( printUsage )
         this->printUsage( programName );
      std::cerr << "Add the following missing  parameters to the command line: " << std::endl << "   ";
      for( int i = 0; i < missingParameters.getSize(); i++ )
         std::cerr << "--" << missingParameters[ i ] << " ... ";
      std::cerr << std::endl;
      return false;
   }
   return true;
}

} // namespace Config
} // namespace TNL

