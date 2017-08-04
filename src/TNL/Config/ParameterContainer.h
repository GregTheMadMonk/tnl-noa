/***************************************************************************
                          Config::ParameterContainer.h  -  description
                             -------------------
    begin                : 2007/06/15
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/Containers/List.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/mpi-supp.h>
#include <TNL/param-types.h>
//#include <TNL/Debugging/StackBacktrace.h>

namespace TNL {
namespace Config {   

struct tnlParameterBase
{
   tnlParameterBase( const String& _name,
                     const String& _type )
   : name( _name ), type( _type ){};
 
   String name, type;

};

template< class T > struct tnlParameter : public tnlParameterBase
{
   tnlParameter( const String& _name,
                 const String& _type,
                 const T& val )
   : tnlParameterBase( _name, _type ), value( val ){};

   T value;
};

//template< class T > const char* getType( const T& val );

class ParameterContainer
{
   public:

   ParameterContainer();

   template< class T > bool addParameter( const String& name,
                                          const T& value );

   bool addParameter( const String& name,
                      const String& value );

   bool checkParameter( const String& name ) const;

   template< class T > bool setParameter( const String& name,
                                          const T& value );

   bool setParameter( const String& name,
                      const String& value );

   template< class T > bool getParameter( const String& name,
                                          T& value,
                                          bool verbose = true ) const
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
      {
         std::cerr << "Missing parameter '" << name << "'." << std::endl;
         throw(0); //PrintStackBacktrace;
      }
      return false;
   }

   template< class T > const T& getParameter( const String& name ) const
   {
      int i;
      const int size = parameters. getSize();
      for( i = 0; i < size; i ++ )
         if( parameters[ i ] -> name == name )
            return ( ( tnlParameter< T >* ) parameters[ i ] ) -> value;
      std::cerr << "The program attempts to get unknown parameter " << name << std::endl;
      std::cerr << "Aborting the program." << std::endl;
      abort();
   }
 
   //! Broadcast to other nodes in MPI cluster
  // void MPIBcast( int root, MPI_Comm mpi_comm = MPI_COMM_WORLD );

   ~ParameterContainer();

   protected:

   Containers::List< tnlParameterBase* > parameters;

};

bool parseCommandLine( int argc, char* argv[],
                       const ConfigDescription& config_description,
                       ParameterContainer& parameters,
                       bool printUsage = true );

template< class T >
bool
ParameterContainer::
addParameter( const String& name, const T& value )
{
   return parameters. Append( new tnlParameter< T >( name, TNL::getType< T >(). getString(), value ) );
};

template< class T >
bool
ParameterContainer::
setParameter( const String& name,
              const T& value )
{
   int i;
   for( i = 0; i < parameters. getSize(); i ++ )
   {
      if( parameters[ i ] -> name == name )
      {
         if( parameters[ i ] -> type == TNL::getType< T >() )
         {
            ( ( tnlParameter< T > * ) parameters[ i ] ) -> value = value;
            return true;
         }
         else
         {
            std::cerr << "Parameter " << name << " already exists with different type "
                 << parameters[ i ] -> type << " not "
                 << TNL::getType< T >() << std::endl;
            abort( );
            return false;
         }
      }
   }
   return addParameter< T >( name, value );
};

} // namespace Config
} // namespace TNL
