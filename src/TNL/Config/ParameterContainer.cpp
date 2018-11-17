/***************************************************************************
                          Config::ParameterContainer.cpp  -  description
                             -------------------
    begin                : 2007/06/15
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <ctype.h>
#include <cstring>
#include <stdio.h>

#include "ParameterContainer.h"
#include <TNL/Object.h>

namespace TNL {
namespace Config {    

bool matob( const char* value, bool& ret_val )
{
   if( strcasecmp( value, "yes" ) == 0 ||
       strcasecmp( value, "true" ) == 0  )
   {
      ret_val =  true;
      return true;
   }
   if( strcasecmp( value, "no" ) == 0 ||
       strcasecmp( value, "false" ) == 0  )
   {
      ret_val = false;
      return true;
   }
   return false;
}

bool
Config::ParameterContainer::
checkParameter( const String& name ) const
{
   const int size = parameters.size();
   for( int i = 0; i < size; i++ )
      if( parameters[ i ]->name == name )
         return true;
   return false;
}

/*void ParameterContainer::MPIBcast( int root, MPI_Comm mpi_comm )
{
#ifdef USE_MPI
   int i;
   int size = parameters. getSize();
   :: MPIBcast( size, 1, root, mpi_comm );
   for( i = 0; i < size; i ++ )
   {
      if( MPIGetRank() == root )
      {
         tnlParameterBase* param = parameters[ i ];
         param -> type. MPIBcast( root, MPI_COMM_WORLD );
         param -> name. MPIBcast( root, MPI_COMM_WORLD );
         if( param -> type == "String" )
         {
            ( ( tnlParameter< String >* ) param ) -> value. MPIBcast( root, mpi_comm );
         }
         if( param -> type == "bool" )
         {
            :: MPIBcast( ( ( tnlParameter< bool >* ) param ) -> value, 1, root, mpi_comm );
         }
         if( param -> type == "int" )
         {
            :: MPIBcast( ( ( tnlParameter< int >* ) param ) -> value, 1, root, mpi_comm );
         }
         if( param -> type == "double" )
         {
            :: MPIBcast( ( ( tnlParameter< double >* ) param ) -> value, 1, root, mpi_comm );
         }
      }
      else
      {
         String param_type, param_name;
         param_type. MPIBcast( root, MPI_COMM_WORLD );
         param_name. MPIBcast( root, MPI_COMM_WORLD );
         if( param_type == "mString" )
         {
            String val;
            val. MPIBcast( root, mpi_comm );
            addParameter< String >( param_name. getString(),
                                     val );
         }
         if( param_type == "bool" )
         {
            bool val;
            :: MPIBcast( val, 1, root, mpi_comm );
            addParameter< bool >( param_name. getString(),
                                  val );
         }
         if( param_type == "int" )
         {
            int val;
            :: MPIBcast( val, 1, root, mpi_comm );
            addParameter< int >( param_name. getString(),
                                 val );
         }
         if( param_type == "double" )
         {
            double val;
            :: MPIBcast( val, 1, root, mpi_comm );
            addParameter< double >( param_name. getString(),
                                    val );
         }

      }
   }
#endif
}
*/


bool
parseCommandLine( int argc, char* argv[],
                  const Config::ConfigDescription& config_description,
                  Config::ParameterContainer& parameters,
                  bool printUsage )
{
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
                  bool bool_val;
                  if( ! matob( value, bool_val ) )
                  {
                     std::cerr << "Yes/true or no/false is required for the parameter " << option << "." << std::endl;
                     parse_error = true;
                  }
                  else bool_list.push_back( bool_val );
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
               bool bool_val;
               if( ! matob( value, bool_val ) )
               {
                  std::cerr << "Yes/true or no/false is required for the parameter " << option << "." << std::endl;
                  parse_error = true;
               }
               else parameters.addParameter< bool >( option, bool_val );
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
