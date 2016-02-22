/***************************************************************************
                          tnlMeshTypeResolver_impl.h  -  description
                             -------------------
    begin                : Nov 28, 2013
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

#ifndef TNLMESHTYPERESOLVER_IMPL_H_
#define TNLMESHTYPERESOLVER_IMPL_H_

#include <core/tnlString.h>
#include <mesh/tnlGrid.h>
#include <mesh/tnlDummyMesh.h>
#include <solvers/tnlSolverStarter.h>

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename ConfigTag,
          bool MeshTypeSupported = tnlConfigTagMesh< ConfigTag, MeshType >::enabled >
class tnlMeshResolverTerminator{};


template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
bool tnlMeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, false  >::run( const tnlParameterContainer& parameters )
{
   return ProblemSetter< Real,
                         Device,
                         Index,
                         tnlDummyMesh< Real, Device, Index >,
                         ConfigTag,
                         tnlSolverStarter< ConfigTag > >::template run< Real, Device, Index, ConfigTag >( parameters );
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
bool tnlMeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::run( const tnlParameterContainer& parameters )
{
   const tnlString& meshFileName = parameters.getParameter< tnlString >( "mesh" );

   tnlString meshType;
   if( ! getObjectType( meshFileName, meshType ) )
   {
      cerr << "I am not able to detect the mesh type from the file " << meshFileName << "." << endl;
      return EXIT_FAILURE;
   }
   cout << meshType << " detected in " << meshFileName << " file." << endl;
   tnlList< tnlString > parsedMeshType;
   if( ! parseObjectType( meshType, parsedMeshType ) )
   {
      cerr << "Unable to parse the mesh type " << meshType << "." << endl;
      return false;
   }
   return resolveMeshDimensions( parameters, parsedMeshType );
}

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
bool tnlMeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::resolveMeshDimensions( const tnlParameterContainer& parameters,
                                                                                                        const tnlList< tnlString >& parsedMeshType )
{
   int dimensions = atoi( parsedMeshType[ 1 ].getString() );

   if( dimensions == 1 )
      return resolveMeshRealType< 1 >( parameters, parsedMeshType );
   if( dimensions == 2 )
      return resolveMeshRealType< 2 >( parameters, parsedMeshType );
   if( dimensions == 3 )
      return resolveMeshRealType< 3 >( parameters, parsedMeshType );
   cerr << "Dimensions higher than 3 are not supported." << endl;
   return false;
}

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
   template< int MeshDimensions, typename, typename >
bool tnlMeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::resolveMeshRealType( const tnlParameterContainer& parameters,
                                                                                                      const tnlList< tnlString >& parsedMeshType )
{
   cerr << "Mesh dimension " << MeshDimensions << " is not supported." << endl;
   return false;
}

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
   template< int MeshDimensions, typename >
bool tnlMeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::resolveMeshRealType( const tnlParameterContainer& parameters,
                                                                                                      const tnlList< tnlString >& parsedMeshType )
{
   if( parsedMeshType[ 2 ] == "float" )
      return resolveMeshIndexType< MeshDimensions, float >( parameters, parsedMeshType );
   if( parsedMeshType[ 2 ] == "double" )
      return resolveMeshIndexType< MeshDimensions, double >( parameters, parsedMeshType );
   if( parsedMeshType[ 2 ] == "long-double" )
      return resolveMeshIndexType< MeshDimensions, long double >( parameters, parsedMeshType );
   cerr << "The type '" << parsedMeshType[ 2 ] << "' is not allowed for real type." << endl;
   return false;
}

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
   template< int MeshDimensions,
             typename MeshRealType,
             typename, typename >
bool tnlMeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::resolveMeshIndexType( const tnlParameterContainer& parameters,
                                                                                                        const tnlList< tnlString >& parsedMeshType )
{
   cerr << "The type '" << parsedMeshType[ 4 ] << "' is not allowed for real type." << endl;
   return false;
}

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
   template< int MeshDimensions,
             typename MeshRealType,
             typename >
bool tnlMeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::resolveMeshIndexType( const tnlParameterContainer& parameters,
                                                                                                        const tnlList< tnlString >& parsedMeshType )
{
   if( parsedMeshType[ 4 ] == "short int" )
      return resolveMeshType< MeshDimensions, MeshRealType, short int >( parameters, parsedMeshType );
   if( parsedMeshType[ 4 ] == "int" )
      return resolveMeshType< MeshDimensions, MeshRealType, int >( parameters, parsedMeshType );
   if( parsedMeshType[ 4 ] == "long int" )
      return resolveMeshType< MeshDimensions, MeshRealType, long int >( parameters, parsedMeshType );
   cerr << "The type '" << parsedMeshType[ 4 ] << "' is not allowed for indexing type." << endl;
   return false;
}

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
   template< int MeshDimensions,
             typename MeshRealType,
             typename MeshIndexType,
             typename, typename >
bool tnlMeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::resolveMeshType( const tnlParameterContainer& parameters,
                                                                                                   const tnlList< tnlString >& parsedMeshType )
{
   cerr << "The type '" << parsedMeshType[ 4 ] << "' is not allowed for indexing type." << endl;
   return false;
}

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
   template< int MeshDimensions,
             typename MeshRealType,
             typename MeshIndexType,
             typename >
bool tnlMeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::resolveMeshType( const tnlParameterContainer& parameters,
                                                                                                   const tnlList< tnlString >& parsedMeshType )
{
   if( parsedMeshType[ 0 ] == "tnlGrid" )
   {
      typedef tnlGrid< MeshDimensions, MeshRealType, Device, MeshIndexType > MeshType;
      return tnlMeshResolverTerminator< ProblemSetter, Real, Device, Index, MeshType, ConfigTag >::run( parameters );
   }
   cerr << "Unknown mesh type " << parsedMeshType[ 0 ] << "." << endl;
   return false;
}

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename ConfigTag >
class tnlMeshResolverTerminator< ProblemSetter, Real, Device, Index, MeshType, ConfigTag, false >
{
   public:
      static bool run( const tnlParameterContainer& parameters )
      {
         cerr << "The mesh type " << ::getType< MeshType >() << " is not supported." << endl;
         return false;
      };
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename ConfigTag >
class tnlMeshResolverTerminator< ProblemSetter, Real, Device, Index, MeshType, ConfigTag, true >
{
   public:
      static bool run( const tnlParameterContainer& parameters )
      {
         return ProblemSetter< Real, Device, Index, MeshType, ConfigTag, tnlSolverStarter< ConfigTag > >::run( parameters );
      }
};

#endif /* TNLMESHTYPERESOLVER_IMPL_H_ */
