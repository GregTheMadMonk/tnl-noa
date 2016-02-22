/***************************************************************************
                          tnlSolverInitiator_impl.h  -  description
                             -------------------
    begin                : Feb 23, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#include <config/tnlParameterContainer.h>
#include <solvers/tnlMeshTypeResolver.h>
#include <solvers/tnlBuildConfigTags.h>
#include <solvers/linear/stationary/tnlSORSolver.h>
#include <solvers/linear/krylov/tnlCGSolver.h>
#include <solvers/linear/krylov/tnlBICGStabSolver.h>
#include <solvers/linear/krylov/tnlGMRESSolver.h>
#include <core/tnlHost.h>
#include <core/tnlCuda.h>

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename ConfigTag,
          bool enabled = tnlConfigTagReal< ConfigTag, Real >::enabled >
class tnlSolverInitiatorRealResolver{};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename ConfigTag,
          bool enabled = tnlConfigTagDevice< ConfigTag, Device >::enabled >
class tnlSolverInitiatorDeviceResolver{};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag,
          bool enabled = tnlConfigTagIndex< ConfigTag, Index >::enabled >
class tnlSolverInitiatorIndexResolver{};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename ConfigTag  >
bool tnlSolverInitiator< ProblemSetter, ConfigTag > :: run( const tnlParameterContainer& parameters )
{
   const tnlString& realType = parameters. getParameter< tnlString >( "real-type" );
   if( parameters. getParameter< int >( "verbose" ) )
      cout << "Setting RealType to   ... " << realType << endl;
   if( realType == "float" )
      return tnlSolverInitiatorRealResolver< ProblemSetter, float, ConfigTag >::run( parameters );
   if( realType == "double" )
      return tnlSolverInitiatorRealResolver< ProblemSetter, double, ConfigTag >::run( parameters );
   if( realType == "long-double" )
      return tnlSolverInitiatorRealResolver< ProblemSetter, long double, ConfigTag >::run( parameters );
   cerr << "The real type '" << realType << "' is not defined. " << endl;
   return false;
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename ConfigTag >
class tnlSolverInitiatorRealResolver< ProblemSetter, Real, ConfigTag, true >
{
   public:
      static bool run( const tnlParameterContainer& parameters )
      {
         const tnlString& device = parameters. getParameter< tnlString >( "device" );
         if( parameters. getParameter< int >( "verbose" ) )
            cout << "Setting DeviceType to ... " << device << endl;

         if( device == "host" )
            return tnlSolverInitiatorDeviceResolver< ProblemSetter, Real, tnlHost, ConfigTag >::run( parameters );
         if( device == "cuda" )
            return tnlSolverInitiatorDeviceResolver< ProblemSetter, Real, tnlCuda, ConfigTag >::run( parameters );
         cerr << "The device '" << device << "' is not defined. " << endl;
         return false;
      }
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename ConfigTag >
class tnlSolverInitiatorRealResolver< ProblemSetter, Real, ConfigTag, false >
{
   public:
      static bool run( const tnlParameterContainer& parameters )
      {
         cerr << "The real type " << parameters.getParameter< tnlString >( "real-type" ) << " is not supported." << endl;
         return false;
      }
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename ConfigTag >
class tnlSolverInitiatorDeviceResolver< ProblemSetter, Real, Device, ConfigTag, true >
{
   public:
      static bool run( const tnlParameterContainer& parameters )
      {
         const tnlString& indexType = parameters. getParameter< tnlString >( "index-type" );
         if( parameters. getParameter< int >( "verbose" ) )
            cout << "Setting IndexType to  ... " << indexType << endl;
         if( indexType == "short-int" )
            return tnlSolverInitiatorIndexResolver< ProblemSetter, Real, Device, short int, ConfigTag >::run( parameters );
         if( indexType == "int" )
            return tnlSolverInitiatorIndexResolver< ProblemSetter, Real, Device, int, ConfigTag >::run( parameters );
         if( indexType == "long int" )
            return tnlSolverInitiatorIndexResolver< ProblemSetter, Real, Device, long int, ConfigTag >::run( parameters );
         cerr << "The index type '" << indexType << "' is not defined. " << endl;
         return false;
      }
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename ConfigTag >
class tnlSolverInitiatorDeviceResolver< ProblemSetter, Real, Device, ConfigTag, false >
{
   public:
      static bool run( const tnlParameterContainer& parameters )
      {
         cerr << "The device " << parameters.getParameter< tnlString >( "device" ) << " is not supported." << endl;
         return false;
      }
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
class tnlSolverInitiatorIndexResolver< ProblemSetter, Real, Device, Index, ConfigTag, false >
{
   public:
      static bool run( const tnlParameterContainer& parameters )
      {
         cerr << "The index " << parameters.getParameter< tnlString >( "index-type" ) << " is not supported." << endl;
         return false;
      }
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
class tnlSolverInitiatorIndexResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >
{
   public:
      static bool run( const tnlParameterContainer& parameters )
      {
         return tnlMeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag >::run( parameters );
      }
};


