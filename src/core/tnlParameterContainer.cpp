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
   return parameters. Append( new tnlParameter< tnlString >( name, GetParameterType( tnlString() ). Data(), tnlString( value ) ) );
}
//--------------------------------------------------------------------------
bool tnlParameterContainer :: SetParameter( const char* name,
                                          const char* value )
{
   int i;
   for( i = 0; i < parameters. Size(); i ++ )
   {
      if( parameters[ i ] -> name == name )
      {
         if( GetParameterType( parameters[ i ] ) == GetParameterType( tnlString() ) ) 
         {
            ( ( tnlParameter< tnlString > * ) parameters[ i ] ) -> value. SetString( value );
            return true;
         }
         else
         {
            cerr << "Parameter " << name << " already exists with different type " 
                 << GetParameterType( parameters[ i ] ) << " not " 
                 << GetParameterType( value ) << endl;
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
   const int parameters_num = parameters. Size();
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
   int size = parameters. Size();
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
            AddParameter< tnlString >( param_name. Data(),
                                     val );           
         }
         if( param_type == "bool" )
         {
            bool val;
            :: MPIBcast( val, 1, root, mpi_comm );
            AddParameter< bool >( param_name. Data(),
                                  val );           
         }
         if( param_type == "int" )
         {
            int val;
            :: MPIBcast( val, 1, root, mpi_comm );
            AddParameter< int >( param_name. Data(),
                                 val );           
         }
         if( param_type == "double" )
         {
            double val;
            :: MPIBcast( val, 1, root, mpi_comm );
            AddParameter< double >( param_name. Data(),
                                    val );           
         }

      }
   }
#endif
}
//--------------------------------------------------------------------------
bool ParseCommandLine( int argc, char* argv[], 
                       const tnlConfigDescription& config_description,
                       tnlParameterContainer& parameters )
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
      const tnlConfigEntryType* entry_type;
      if( ! ( entry_type = config_description. GetEntryType( option ) ) )
      {
         cerr << "Unknown parameter " << option << "." << endl;
         parse_error = true;
      }
      else 
      {
         const char* value = argv[ ++ i ];
         if( ! value )
         {
            cerr << "Missing value for the parameter " << option << "." << endl;
            return false;
         }
         const tnlString& basic_type_name = entry_type -> basic_type;
         if( entry_type -> list_entry )
         {
            tnlList< tnlString >* string_list( 0 );
            tnlList< bool >* bool_list( 0 );
            tnlList< int >* integer_list( 0 );
            tnlList< double >* real_list( 0 );

            if( basic_type_name == "string" )
               string_list = new tnlList< tnlString >;
            if( basic_type_name == "bool" )
               bool_list = new tnlList< bool >;
            if( basic_type_name == "integer" )
               integer_list = new tnlList< int >;
            if( basic_type_name == "real" )
               real_list = new tnlList< double >;
            
            while( i < argc && ( ( argv[ i ] )[ 0 ] != '-' || ( atof( argv[ i ] ) < 0.0 && ( integer_list || real_list ) ) ) )
            {
               const char* value = argv[ i ++ ];
               if( string_list ) string_list -> Append( tnlString( value ) );
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
               if( integer_list ) integer_list -> Append( atoi( value ) );
               if( real_list ) real_list -> Append( atof( value ) );
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
            if( basic_type_name == "string" )
            {
                parameters. AddParameter< tnlString >( option, value );
                continue;
            }
            if( basic_type_name == "bool" )
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
            if( basic_type_name == "integer" )
            {
               /*if( ! isdigit( value ) )
               {
                  cerr << "Integer constant is required for the parameter " << option << "." << endl;
                  parse_error = true;
                  continue;
               }*/
               parameters. AddParameter< int >( option, atoi( value ) );
            }
            if( basic_type_name == "real" )
            {
               /*if( ! isdigit( value ) )
               {
                  cerr << "Real constant is required for the parameter " << option << "." << endl;
                  parse_error = true;
                  continue;
               }*/
               parameters. AddParameter< double >( option, atof( value ) );
            }
         }
      }
   }
   config_description. AddMissingEntries( parameters );
   if( ! config_description. CheckMissingEntries( parameters ) )
      return false;
   return ! parse_error;
}
