/***************************************************************************
                          Config::ParameterContainer.h  -  description
                             -------------------
    begin                : 2007/06/15
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <vector>
#include <memory>
#include "make_unique.h"

#include <TNL/param-types.h>
//#include <TNL/Debugging/StackBacktrace.h>

namespace TNL {
namespace Config {

struct ParameterBase
{
   ParameterBase( const String& name,
                  const String& type )
   : name( name ), type( type )
   {}

   String name, type;

   // Virtual destructor is needed to avoid undefined behaviour when deleting the
   // ParameterContainer::parameters vector, see https://stackoverflow.com/a/8752126
   virtual ~ParameterBase() = default;
};

template< class T >
struct Parameter : public ParameterBase
{
   Parameter( const String& name,
              const String& type,
              const T& value )
   : ParameterBase( name, type ), value( value )
   {}

   T value;
};

class ParameterContainer
{
public:
   /**
    * \brief Adds new parameter to the ParameterContainer.
    *
    * \tparam T Type of parameter value.
    * \param name Name of the new parameter.
    * \param value Value assigned to the parameter.
    */
   template< class T >
   bool addParameter( const String& name,
                      const T& value )
   {
      parameters.push_back( std::make_unique< Parameter< T > >( name, TNL::getType< T >(), value ) );
      return true;
   }

   /**
    * \brief Checks whether the parameter \e name already exists in ParameterContainer.
    *
    * \param name Name of the parameter.
    */
   bool checkParameter( const String& name ) const
   {
      const int size = parameters.size();
      for( int i = 0; i < size; i++ )
         if( parameters[ i ]->name == name )
            return true;
      return false;
   }

   /**
    * \brief Assigns new \e value to the parameter \e name.
    *
    * \tparam T Type of the parameter value.
    * \param name Name of parameter.
    * \param value Value of type T assigned to the parameter.
    */
   template< class T >
   bool setParameter( const String& name,
                      const T& value )
   {
      for( int i = 0; i < (int) parameters.size(); i++ ) {
         if( parameters[ i ]->name == name ) {
            if( parameters[ i ]->type == TNL::getType< T >() ) {
               Parameter< T >& parameter = dynamic_cast< Parameter< T >& >( *parameters[ i ] );
               parameter.value = value;
               return true;
            }
            else {
               std::cerr << "Parameter " << name << " already exists with different type "
                         << parameters[ i ]->type << " not "
                         << TNL::getType< T >() << std::endl;
               throw 0;
            }
         }
      }
      return addParameter< T >( name, value );
   }

   /**
    * \brief Checks whether the parameter \e name is given the \e value.
    *
    * Returns \e true if the parameter \e name is given the \e value.
    * If the parameter does not have any value or has different value then the given
    * \e value the method returns \e false and shows message when \e verbose is \e true.
    *
    * \param name Name of parameter.
    * \param value Value of type T we want to check whether is assigned to the parameter.
    * \param verbose Boolean value defining whether to show error message (when true) or not (when false).
    *
    * \par Example
    * \include ParameterContainerExample.cpp
    */
   template< class T >
   bool getParameter( const String& name,
                      T& value,
                      bool verbose = true ) const
   {
      for( int i = 0; i < (int) parameters.size(); i++ )
         if( parameters[ i ]->name == name )
         {
            // dynamic_cast throws std::bad_cast if parameters[i] does not have the type Parameter<T>
            const Parameter< T >& parameter = dynamic_cast< Parameter< T >& >( *parameters[ i ] );
            value = parameter.value;
            return true;
         }
      if( verbose )
      {
         std::cerr << "Missing parameter '" << name << "'." << std::endl;
         throw 0; //PrintStackBacktrace;
      }
      return false;
   }

   /**
    * \brief Returns parameter value.
    *
    * \param name Name of parameter.
    */
   template< class T >
   T getParameter( const String& name ) const
   {
      for( int i = 0; i < (int) parameters.size(); i++ )
         if( parameters[ i ]->name == name )
         {
            // dynamic_cast throws std::bad_cast if parameters[i] does not have the type Parameter<T>
            const Parameter< T >& parameter = dynamic_cast< Parameter< T >& >( *parameters[ i ] );
            return parameter.value;
         }
      std::cerr << "The program attempts to get unknown parameter " << name << std::endl;
      std::cerr << "Aborting the program." << std::endl;
      throw 0;
   }

/*
   //! Broadcast to other nodes in MPI cluster
   void MPIBcast( int root, MPI_Comm mpi_comm = MPI_COMM_WORLD )
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

protected:
   std::vector< std::unique_ptr< ParameterBase > > parameters;
};

} // namespace Config
} // namespace TNL
