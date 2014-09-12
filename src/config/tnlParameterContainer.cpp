/***************************************************************************
                          tnlParameterContainer.cpp  -  description
                             -------------------
    begin                : 2007/06/15
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

#include <ctype.h>
#include <cstring>
#include <stdio.h>
#include "tnlParameterContainer.h"

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
//--------------------------------------------------------------------------
tnlParameterContainer :: tnlParameterContainer()
{
}
//--------------------------------------------------------------------------
bool tnlParameterContainer :: AddParameter( const char* name,
                                            const char* value )
{
   return parameters. Append( new tnlParameter< tnlString >( name, ::getType< tnlString >().getString(), tnlString( value ) ) );
}
//--------------------------------------------------------------------------
bool tnlParameterContainer :: SetParameter( const char* name,
                                            const char* value )
{
   int i;
   for( i = 0; i < parameters. getSize(); i ++ )
   {
      if( parameters[ i ] -> name == name )
      {
         if( parameters[ i ] -> type == ::getType< tnlString >() )
         {
            ( ( tnlParameter< tnlString > * ) parameters[ i ] ) -> value. setString( value );
            return true;
         }
         else
         {
            cerr << "Parameter " << name << " already exists with different type " 
                 << parameters[ i ] -> type << " not " 
                 << ::getType< tnlString>() << endl;
            abort();
            return false;
         }
      }
   }
   return AddParameter( name, value );
};
//--------------------------------------------------------------------------
bool tnlParameterContainer :: CheckParameter( const char* name ) const
{
   int i;
   const int parameters_num = parameters. getSize();
   for( i = 0; i < parameters_num; i ++ )
      if( parameters[ i ] -> name == name ) return true;
   return false;
}
//--------------------------------------------------------------------------
tnlParameterContainer :: ~tnlParameterContainer()
{
   parameters. DeepEraseAll();
}
//--------------------------------------------------------------------------
void tnlParameterContainer :: MPIBcast( int root, MPI_Comm mpi_comm )
{
#ifdef HAVE_MPI
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
         if( param -> type == "mString" )
         {
            ( ( tnlParameter< tnlString >* ) param ) -> value. MPIBcast( root, mpi_comm );
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
         tnlString param_type, param_name;
         param_type. MPIBcast( root, MPI_COMM_WORLD );
         param_name. MPIBcast( root, MPI_COMM_WORLD );
         if( param_type == "mString" )
         {
            tnlString val;
            val. MPIBcast( root, mpi_comm );
            AddParameter< tnlString >( param_name. getString(),
                                     val );           
         }
         if( param_type == "bool" )
         {
            bool val;
            :: MPIBcast( val, 1, root, mpi_comm );
            AddParameter< bool >( param_name. getString(),
                                  val );           
         }
         if( param_type == "int" )
         {
            int val;
            :: MPIBcast( val, 1, root, mpi_comm );
            AddParameter< int >( param_name. getString(),
                                 val );           
         }
         if( param_type == "double" )
         {
            double val;
            :: MPIBcast( val, 1, root, mpi_comm );
            AddParameter< double >( param_name. getString(),
                                    val );           
         }

      }
   }
#endif
}
//--------------------------------------------------------------------------
bool ParseCommandLine( int argc, char* argv[], 
                       const tnlConfigDescription& config_description,
                       tnlParameterContainer& parameters,
                       bool printUsage )
{
   int i;
   bool parse_error( false );
   for( i = 1; i < argc; i ++ )
   {
      const char* _option = argv[ i ];
      if( _option[ 0 ] != '-' )
      {
         cerr << "Unknown option " << _option << ". Options must have prefix '--' or '-'." << endl;
         parse_error = true;
         continue;
      }
      
      const char* option = _option + 2;
      const tnlConfigEntryBase* entry;
      if( ( entry = config_description.getEntry( option ) ) == NULL )
      {
         cerr << "Unknown parameter " << _option << "." << endl;
         parse_error = true;
      }
      else 
      {
         const tnlString& entryType = entry->getEntryType();
         const char* value = argv[ ++ i ];
         if( ! value )
         {
            cerr << "Missing value for the parameter " << option << "." << endl;
            return false;
         }
         tnlList< tnlString > parsedEntryType;
         if( ! parseObjectType( entryType, parsedEntryType ) )
         {
            cerr << "Internal error: Uknown config entry type " << entryType << "." << endl;
            return false;
         }
         if( parsedEntryType[ 0 ] == "tnlList" )
         {
            tnlList< tnlString >* string_list( 0 );
            tnlList< bool >* bool_list( 0 );
            tnlList< int >* integer_list( 0 );
            tnlList< double >* real_list( 0 );

            if( parsedEntryType[ 1 ] == "tnlString" )
               string_list = new tnlList< tnlString >;
            if( parsedEntryType[ 1 ] == "bool" )
               bool_list = new tnlList< bool >;
            if( parsedEntryType[ 1 ] == "int" )
               integer_list = new tnlList< int >;
            if( parsedEntryType[ 1 ] == "double" )
               real_list = new tnlList< double >;
            
            while( i < argc && ( ( argv[ i ] )[ 0 ] != '-' || ( atof( argv[ i ] ) < 0.0 && ( integer_list || real_list ) ) ) )
            {
               const char* value = argv[ i ++ ];
               if( string_list )
               {
                  /*if( ! ( ( tnlConfigEntry< tnlList< tnlString > >* )  entry )->checkValue( tnlString( value ) ) )
                  {
                     delete string_list;
                     return false;
                  }*/
                  string_list -> Append( tnlString( value ) );
               }
               if( bool_list )
               {
                  bool bool_val;
                  if( ! matob( value, bool_val ) )
                  {
                     cerr << "Yes/true or no/false is required for the parameter " << option << "." << endl;
                     parse_error = true;
                  }
                  else bool_list -> Append( bool_val );
               }
               if( integer_list )
               {
                  /*if( ! ( tnlConfigEntry< tnlList< int > >* ) entry->checkValue( atoi( value ) ) )
                  {
                     delete integer_list;
                     return false;
                  }*/
                  integer_list -> Append( atoi( value ) );
               }
               if( real_list )
               {
                  /*if( ! ( tnlConfigEntry< tnlList< double > >* ) entry->checkValue( atof( value ) ) )
                  {
                     delete real_list;
                     return false;
                  }*/
                  real_list -> Append( atof( value ) );
               }
            }
            if( string_list )
            {
               parameters. AddParameter< tnlList< tnlString > >( option, *string_list );
               delete string_list;
            }
            if( bool_list )
            {
               parameters. AddParameter< tnlList< bool > >( option, *bool_list );
               delete bool_list;
            }
            if( integer_list )
            {
               parameters. AddParameter< tnlList< int > >( option, *integer_list );
               delete integer_list;
            }
            if( real_list )
            {
               parameters. AddParameter< tnlList< double > >( option, *real_list );
               delete real_list;
            }
            if( i < argc ) i --;
            continue;
         }
         else
         {
            if( parsedEntryType[ 0 ] == "tnlString" )
            {
               if( ! ( ( tnlConfigEntry< tnlString >* ) entry )->checkValue( value ) )
                  return false;
                parameters. AddParameter< tnlString >( option, value );
                continue;
            }
            if( parsedEntryType[ 0 ] == "bool" )
            {
               bool bool_val;
               if( ! matob( value, bool_val ) )
               {
                  cerr << "Yes/true or no/false is required for the parameter " << option << "." << endl;
                  parse_error = true;
               }
               else parameters. AddParameter< bool >( option, bool_val );
               continue;
            }
            if( parsedEntryType[ 0 ] == "int" )
            {
               /*if( ! isdigit( value ) )
               {
                  cerr << "Integer constant is required for the parameter " << option << "." << endl;
                  parse_error = true;
                  continue;
               }*/
               if( ! ( ( tnlConfigEntry< int >* ) entry )->checkValue( atoi( value ) ) )
                  return false;
               parameters. AddParameter< int >( option, atoi( value ) );
            }
            if( parsedEntryType[ 0 ] == "double" )
            {
               /*if( ! isdigit( value ) )
               {
                  cerr << "Real constant is required for the parameter " << option << "." << endl;
                  parse_error = true;
                  continue;
               }*/
               if( ! ( ( tnlConfigEntry< double >* ) entry )->checkValue( atof( value ) ) )
                  return false;
               parameters. AddParameter< double >( option, atof( value ) );
            }
         }
      }
   }
   config_description.addMissingEntries( parameters );
   if( ! config_description.checkMissingEntries( parameters, printUsage, argv[ 0 ] ) )
      return false;
   return ! parse_error;
}
