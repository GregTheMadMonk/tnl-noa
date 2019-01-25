/***************************************************************************
                          parseCommandLine.h  -  description
                             -------------------
    begin                : 2007/06/15
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <cstring>
#include <string>

//#include <TNL/Object.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>

namespace TNL {

std::vector< String >
parseObjectType( const String& objectType );

namespace Config {

inline bool
parseCommandLine( int argc, char* argv[],
                  const Config::ConfigDescription& config_description,
                  Config::ParameterContainer& parameters,
                  bool printUsage = true )
{
   auto iequals = []( const std::string& a, const std::string& b )
   {
      if( a.size() != b.size() )
         return false;
      for( unsigned int i = 0; i < a.size(); i++ )
         if( std::tolower(a[i]) != std::tolower(b[i]) )
            return false;
      return true;
   };

   auto matob = [iequals]( const char* value )
   {
      if( iequals( value, "yes" ) || iequals( value, "true" ) )
         return true;
      if( iequals( value, "no" ) || iequals( value, "false" ) )
         return true;
      return false;
   };

   int i;
   bool parse_error( false );
   for( i = 1; i < argc; i ++ )
   {
      const char* _option = argv[ i ];
      if( _option[ 0 ] != '-' )
      {
         std::cerr << "Unknown option " << _option << ". Options must have prefix '--' or '-'." << std::endl;
         parse_error = true;
         continue;
      }
      if( strcmp( _option, "--help" ) == 0 )
      {
          config_description.printUsage( argv[ 0 ] );
          return true;
      }
      const char* option = _option + 2;
      const ConfigEntryBase* entry;
      if( ( entry = config_description.getEntry( option ) ) == NULL )
      {
         std::cerr << "Unknown parameter " << _option << "." << std::endl;
         parse_error = true;
      }
      else
      {
         const String& entryType = entry->getEntryType();
         const char* value = argv[ ++ i ];
         if( ! value )
         {
            std::cerr << "Missing value for the parameter " << option << "." << std::endl;
            return false;
         }
         std::vector< String > parsedEntryType = parseObjectType( entryType );
         if( parsedEntryType.size() == 0 )
         {
            std::cerr << "Internal error: Unknown config entry type " << entryType << "." << std::endl;
            return false;
         }
         if( parsedEntryType[ 0 ] == "ConfigEntryList" )
         {
            std::vector< String > string_list;
            std::vector< bool > bool_list;
            std::vector< int > integer_list;
            std::vector< double > real_list;

            while( i < argc && ( ( argv[ i ] )[ 0 ] != '-' || ( atof( argv[ i ] ) < 0.0 && ( parsedEntryType[ 1 ] == "int" || parsedEntryType[ 1 ] == "double" ) ) ) )
            {
               const char* value = argv[ i ++ ];
               if( parsedEntryType[ 1 ] == "String" )
               {
                  string_list.push_back( String( value ) );
               }
               if( parsedEntryType[ 1 ] == "bool" )
               {
                  if( ! matob( value ) )
                  {
                     std::cerr << "Yes/true or no/false is required for the parameter " << option << "." << std::endl;
                     parse_error = true;
                  }
                  else bool_list.push_back( true );
               }
               if( parsedEntryType[ 1 ] == "int" )
               {
                  integer_list.push_back( atoi( value ) );
               }
               if( parsedEntryType[ 1 ] == "double" )
               {
                  real_list.push_back( atof( value ) );
               }
            }
            if( string_list.size() )
               parameters.addParameter< std::vector< String > >( option, string_list );
            if( bool_list.size() )
               parameters.addParameter< std::vector< bool > >( option, bool_list );
            if( integer_list.size() )
               parameters.addParameter< std::vector< int > >( option, integer_list );
            if( real_list.size() )
               parameters.addParameter< std::vector< double > >( option, real_list );
            if( i < argc ) i --;
            continue;
         }
         else
         {
            if( parsedEntryType[ 0 ] == "String" )
            {
               if( ! ( ( ConfigEntry< String >* ) entry )->checkValue( value ) )
                  return false;
                parameters.addParameter< String >( option, value );
                continue;
            }
            if( parsedEntryType[ 0 ] == "bool" )
            {
               if( ! matob( value ) )
               {
                  std::cerr << "Yes/true or no/false is required for the parameter " << option << "." << std::endl;
                  parse_error = true;
               }
               else parameters.addParameter< bool >( option, true );
               continue;
            }
            if( parsedEntryType[ 0 ] == "int" )
            {
               /*if( ! std::isdigit( value ) ) //TODO: Check for real number
               {
                  std::cerr << "Integer constant is required for the parameter " << option << "." << std::endl;
                  parse_error = true;
                  continue;
               }*/
               if( ! ( ( ConfigEntry< int >* ) entry )->checkValue( atoi( value ) ) )
                  return false;
               parameters.addParameter< int >( option, atoi( value ) );
            }
            if( parsedEntryType[ 0 ] == "double" )
            {
               /*if( ! std::isdigit( value ) )  //TODO: Check for real number
               {
                  std::cerr << "Real constant is required for the parameter " << option << "." << std::endl;
                  parse_error = true;
                  continue;
               }*/
               if( ! ( ( ConfigEntry< double >* ) entry )->checkValue( atof( value ) ) )
                  return false;
               parameters.addParameter< double >( option, atof( value ) );
            }
         }
      }
   }
   config_description.addMissingEntries( parameters );
   if( ! config_description.checkMissingEntries( parameters, printUsage, argv[ 0 ] ) )
      return false;
   return ! parse_error;
}

} // namespace Config
} // namespace TNL
