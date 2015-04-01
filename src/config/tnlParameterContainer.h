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

#include <core/tnlList.h>
#include <config/tnlConfigDescription.h>
#include <core/mpi-supp.h>
#include <core/param-types.h>

struct tnlParameterBase
{
   tnlParameterBase( const tnlString& _name,
                     const tnlString& _type )
   : name( _name ), type( _type ){};
 
   tnlString name, type;

};

template< class T > struct tnlParameter : public tnlParameterBase
{
   tnlParameter( const tnlString& _name,
                 const tnlString& _type,
                 const T& val )
   : tnlParameterBase( _name, _type ), value( val ){};

   T value;
};

//template< class T > const char* getType( const T& val );

class tnlParameterContainer
{
   public:

   tnlParameterContainer();

   template< class T > bool addParameter( const tnlString& name,
                                          const T& value );

   bool addParameter( const tnlString& name,
                      const tnlString& value );

   bool checkParameter( const tnlString& name ) const;

   template< class T > bool setParameter( const tnlString& name,
                                          const T& value );

   bool setParameter( const tnlString& name,
                      const tnlString& value );

   template< class T > bool getParameter( const tnlString& name,
                                          T& value,
                                          bool verbose = false ) const
   {
      int i;
      const int size = parameters. getSize();
      for( i = 0; i < size; i ++ )
         if( parameters[ i ] -> name == name )
         {
            value = ( ( tnlParameter< T >* ) parameters[ i ] ) -> value;
            return true;
         }
      if( verbose )
         cerr << "Missing parameter '" << name << "'." << endl;
      return false;
   }

   template< class T > const T& getParameter( const tnlString& name ) const
   {
      int i;
      const int size = parameters. getSize();
      for( i = 0; i < size; i ++ )
         if( parameters[ i ] -> name == name )
            return ( ( tnlParameter< T >* ) parameters[ i ] ) -> value;
      cerr << "The program attempts to get unknown parameter " << name << endl;
      cerr << "Aborting the program." << endl;
      abort();
   }
   
   //! Broadcast to other nodes in MPI cluster
   void MPIBcast( int root, MPI_Comm mpi_comm = MPI_COMM_WORLD );

   ~tnlParameterContainer();

   protected:

   tnlList< tnlParameterBase* > parameters;

};

bool parseCommandLine( int argc, char* argv[],
                       const tnlConfigDescription& config_description,
                       tnlParameterContainer& parameters,
                       bool printUsage = true );

template< class T >
bool
tnlParameterContainer::
addParameter( const tnlString& name, const T& value )
{
   return parameters. Append( new tnlParameter< T >( name, ::getType< T >(). getString(), value ) );
};

template< class T >
bool
tnlParameterContainer::
setParameter( const tnlString& name,
              const T& value )
{
   int i;
   for( i = 0; i < parameters. getSize(); i ++ )
   {
      if( parameters[ i ] -> name == name )
      {
         if( parameters[ i ] -> type == getType( value ) )
         {
            ( ( tnlParameter< T > * ) parameters[ i ] ) -> value = value;
            return true;
         }
         else
         {
            cerr << "Parameter " << name << " already exists with different type " 
                 << parameters[ i ] -> type << " not "
                 << getType( value ) << endl;
            abort( ); 
            return false;
         }
      }
   }
   return addParameter< T >( name, value );
};
#endif
