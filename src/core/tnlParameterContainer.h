/***************************************************************************
                          tnlParameterContainer.h  -  description
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

#ifndef tnlParameterContainerH
#define tnlParameterContainerH

#include "tnlList.h"
#include "tnlConfigDescription.h"
#include "mpi-supp.h"
#include "param-types.h"

struct tnlParameterBase
{
   tnlParameterBase( const char* _name, const char* _type )
   : name( _name ), type( _type ){};
 
   tnlString name, type;

};

template< class T > struct tnlParameter : public tnlParameterBase
{
   tnlParameter( const char* _name,
               const char* _type,
               const T& val )
   : tnlParameterBase( _name, _type ), value( val ){};

   T value;
};

//template< class T > const char* GetParameterType( const T& val );

class tnlParameterContainer
{
   public:

   tnlParameterContainer();

   template< class T > bool AddParameter( const char* name,
                                          const T& value );

   bool AddParameter( const char* name, 
                      const char* value );

   bool CheckParameter( const char* name ) const;

   template< class T > bool SetParameter( const char* name,
                                          const T& value );

   bool SetParameter( const char* name,
                      const char* value );

   template< class T > bool GetParameter( const char* name, T& value, bool verbose = false ) const
   {
      int i;
      const int size = parameters. Size();
      for( i = 0; i < size; i ++ )
         if( parameters[ i ] -> name == name )
         {
            value = ( ( tnlParameter< T >* ) parameters[ i ] ) -> value;
            return true;
         }
      if( verbose )
         cerr << "Missing parameter '" << name << "'." << endl;
      return false;
   };

   template< class T > const T& GetParameter( const char* name ) const
   {
      int i;
      const int size = parameters. Size();
      for( i = 0; i < size; i ++ )
         if( parameters[ i ] -> name == name )
            return ( ( tnlParameter< T >* ) parameters[ i ] ) -> value;
      cerr << "Unknown parameter " << name << endl;
      abort();
   };
   
   template< class T > T& GetParameter( const char* name )
   {
      int i;
      const int size = parameters. Size();
      for( i = 0; i < size; i ++ )
         if( parameters[ i ] -> name == name )
            return ( ( tnlParameter< T >* ) parameters[ i ] ) -> value;
      cerr << "Unknown parameter " << name << endl;
      abort();
   };
   
   //! Broadcast to other nodes in MPI cluster
   void MPIBcast( int root, MPI_Comm mpi_comm = MPI_COMM_WORLD );

   ~tnlParameterContainer();

   protected:

   tnlList< tnlParameterBase* > parameters;

};

bool ParseCommandLine( int argc, char* argv[], 
                       const tnlConfigDescription& config_description,
                       tnlParameterContainer& parameters );

template< class T > bool tnlParameterContainer :: AddParameter( const char* name,
                                                                const T& value )
{
   return parameters. Append( new tnlParameter< T >( name, GetParameterType( value ). Data(), value ) );
};

template< class T > bool tnlParameterContainer :: SetParameter( const char* name,
                                                                const T& value )
{
   int i;
   for( i = 0; i < parameters. Size(); i ++ )
   {
      if( parameters[ i ] -> name == name )
      {
         if( parameters[ i ] -> type == GetParameterType( value ) )
         {
            ( ( tnlParameter< T > * ) parameters[ i ] ) -> value = value;
            return true;
         }
         else
         {
            cerr << "Parameter " << name << " already exists with different type " 
                 << parameters[ i ] -> type << " not "
                 << GetParameterType( value ) << endl;
            abort( ); 
            return false;
         }
      }
   }
   return AddParameter< T >( name, value );
};
#endif
