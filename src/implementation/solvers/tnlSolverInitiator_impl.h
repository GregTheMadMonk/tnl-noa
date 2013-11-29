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

template< template< typename MeshType, typename SolverStarter > class ProblemSetter,
          typename SolverConfig  >
bool tnlSolverInitiator< ProblemSetter, SolverConfig > :: run( const char* configFileName, int argc, char* argv[] )
{
   tnlParameterContainer parameters;
   tnlConfigDescription conf_desc;
   if( conf_desc. ParseConfigDescription( configFileName ) != 0 )
      return false;
   if( ! ParseCommandLine( argc, argv, conf_desc, parameters ) )
   {
      conf_desc. PrintUsage( argv[ 0 ] );
      return false;
   }
   this -> verbose = parameters. GetParameter< int >( "verbose" );
   return setRealType( parameters );
};

template< template< typename MeshType, typename SolverStarter > class ProblemSetter,
          typename SolverConfig  >
bool tnlSolverInitiator< ProblemSetter, SolverConfig > :: checkSupportedRealTypes( const tnlString& realType,
                                                                     const tnlParameterContainer& parameters ) const
{
   return true;
}

template< template< typename MeshType, typename SolverStarter > class ProblemSetter,
          typename SolverConfig >
bool tnlSolverInitiator< ProblemSetter, SolverConfig > :: checkSupportedIndexTypes( const tnlString& indexType,
                                                                      const tnlParameterContainer& parameters ) const
{
   return true;
}

template< template< typename MeshType, typename SolverStarter > class ProblemSetter,
          typename SolverConfig  >
bool tnlSolverInitiator< ProblemSetter, SolverConfig > :: checkSupportedDevices( const tnlString& device,
                                                                   const tnlParameterContainer& parameters ) const
{
   return true;
}

template< template< typename MeshType, typename SolverStarter > class ProblemSetter,
          typename SolverConfig  >
bool tnlSolverInitiator< ProblemSetter, SolverConfig > :: setRealType( const tnlParameterContainer& parameters ) const
{
   const tnlString& realType = parameters. GetParameter< tnlString >( "real-type" );
   if( ! this -> checkSupportedRealTypes( realType, parameters ) )
   {
      cerr << "The real type '" << realType << "' is not supported." << endl;
      return false;
   }
   if( this -> verbose )
      cout << "Setting RealType to   ... " << realType << endl;
   if( realType == "float" )
      return setIndexType< float >( parameters );
   if( realType == "double" )
      return setIndexType< double >( parameters );
   if( realType == "long-double" )
      return setIndexType< long double >( parameters );
   cerr << "The real type '" << realType << "' is not defined. " << endl;
   return false;
}

template< template< typename MeshType, typename SolverStarter > class ProblemSetter,
          typename SolverConfig  >
   template< typename RealType >
bool tnlSolverInitiator< ProblemSetter, SolverConfig > :: setIndexType( const tnlParameterContainer& parameters ) const
{
   const tnlString& indexType = parameters. GetParameter< tnlString >( "index-type" );
   if( ! this -> checkSupportedIndexTypes( indexType, parameters ) )
   {
      cerr << "The index type '" << indexType << "' is not supported." << endl;
      return false;
   }
   if( this -> verbose )
      cout << "Setting IndexType to  ... " << indexType << endl;
   if( indexType == "int" )
      return setDeviceType< RealType, int >( parameters );
   if( indexType == "long int" )
      return setDeviceType< RealType, long int >( parameters );
   cerr << "The index type '" << indexType << "' is not defined. " << endl;
   return false;
}

template< template< typename MeshType, typename SolverStarter > class ProblemSetter,
          typename SolverConfig  >
   template< typename RealType,
             typename IndexType >
bool tnlSolverInitiator< ProblemSetter, SolverConfig > :: setDeviceType( const tnlParameterContainer& parameters ) const
{
   const tnlString& device = parameters. GetParameter< tnlString >( "device" );
   if( ! this -> checkSupportedDevices( device, parameters ) )
   {
      cerr << "The device '" << device << "' is not supported." << endl;
      return false;
   }
   if( this -> verbose )
      cout << "Setting DeviceType to ... " << device << endl;

   if( device == "host" )
      return tnlMeshTypeResolver< SolverConfig::ResolveMesh, RealType, tnlHost, IndexType, ProblemSetter >::run( parameters );
   if( device == "cuda" )
      return tnlMeshTypeResolver< SolverConfig::ResolveMesh, RealType, tnlCuda, IndexType, ProblemSetter >::run( parameters );
   cerr << "The device '" << device << "' is not defined. " << endl;
   return false;
}
