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

template< typename Real,
          typename Device,
          typename Index,
          template< typename MeshType, typename SolverStarter > class ProblemSetter >
bool tnlMeshTypeResolver< false, Real, Device, Index, ProblemSetter >::run( const tnlParameterContainer& parameters )
{
   return ProblemSetter< tnlDummyMesh< Real, Device, Index >,
                         tnlSolverStarter >::template run< Real, Device, Index >( parameters );
}

template< typename Real,
          typename Device,
          typename Index,
          template< typename MeshType, typename SolverStarter > class ProblemSetter >
bool tnlMeshTypeResolver< true, Real, Device, Index, ProblemSetter >::run( const tnlParameterContainer& parameters )
{
   const tnlString& meshFileName = parameters.GetParameter< tnlString >( "mesh" );

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

template< typename Real,
          typename Device,
          typename Index,
          template< typename MeshType, typename SolverStarter > class ProblemSetter >
bool tnlMeshTypeResolver< true, Real, Device, Index, ProblemSetter >::resolveMeshDimensions( const tnlParameterContainer& parameters,
                                                                                             const tnlList< tnlString >& parsedMeshType )
{
   int dimensions = atoi( parsedMeshType[ 1 ].getString() );

   if( dimensions == 1 )
      return tnlMeshTypeResolverDimensionsSupportChecker< 1,
                                                          true, //ProblemConfig::SupportDimensions1,
                                                          tnlMeshTypeResolver< true, Real, Device, Index, ProblemSetter > >::
                                                          checkDimensions( parameters, parsedMeshType );

   if( dimensions == 2 )
      return tnlMeshTypeResolverDimensionsSupportChecker< 2,
                                                          true, //ProblemConfig::SupportDimensions2,
                                                          tnlMeshTypeResolver< true, Real, Device, Index, ProblemSetter > >::
                                                          checkDimensions( parameters, parsedMeshType );
   if( dimensions == 3 )
      return tnlMeshTypeResolverDimensionsSupportChecker< 3,
                                                          true, //ProblemConfig::SupportDimensions3,
                                                          tnlMeshTypeResolver< true, Real, Device, Index, ProblemSetter > >::
                                                          checkDimensions( parameters, parsedMeshType );
}

template< typename Real,
          typename Device,
          typename Index,
          template< typename MeshType, typename SolverStarter > class ProblemSetter >
   template< int MeshDimensions >
bool tnlMeshTypeResolver< true, Real, Device, Index, ProblemSetter >::resolveMeshRealType( const tnlParameterContainer& parameters,
                                                                                           const tnlList< tnlString >& parsedMeshType )
{
   if( parsedMeshType[ 2 ] == "float" )
      return resolveMeshIndexType< MeshDimensions, float >( parameters, parsedMeshType );

   if( parsedMeshType[ 2 ] == "double" )
      return resolveMeshIndexType< MeshDimensions, double >( parameters, parsedMeshType );

   if( parsedMeshType[ 2 ] == "long-double" )
      return resolveMeshIndexType< MeshDimensions, long double >( parameters, parsedMeshType );
}

template< typename Real,
          typename Device,
          typename Index,
          template< typename MeshType, typename SolverStarter > class ProblemSetter >
   template< int MeshDimensions,
             typename MeshRealType >
bool tnlMeshTypeResolver< true, Real, Device, Index, ProblemSetter >::resolveMeshIndexType( const tnlParameterContainer& parameters,
                                                                                            const tnlList< tnlString >& parsedMeshType )
{
   if( parsedMeshType[ 4 ] == "int" )
      return resolveMeshType< MeshDimensions, MeshRealType, int >( parameters, parsedMeshType );

   if( parsedMeshType[ 4 ] == "long int" )
      return resolveMeshType< MeshDimensions, MeshRealType, long int >( parameters, parsedMeshType );
}

template< typename Real,
          typename Device,
          typename Index,
          template< typename MeshType, typename SolverStarter > class ProblemSetter >
   template< int MeshDimensions,
             typename MeshRealType,
             typename MeshIndexType >
bool tnlMeshTypeResolver< true, Real, Device, Index, ProblemSetter >::resolveMeshType( const tnlParameterContainer& parameters,
                                                                                       const tnlList< tnlString >& parsedMeshType )
{
   if( parsedMeshType[ 0 ] == "tnlGrid" )
      return resolveGridGeometryType< MeshDimensions, MeshRealType, MeshIndexType >( parameters, parsedMeshType );
}

template< typename Real,
          typename Device,
          typename Index,
          template< typename MeshType, typename SolverStarter > class ProblemSetter >
   template< int MeshDimensions,
             typename MeshRealType,
             typename MeshIndexType >
bool tnlMeshTypeResolver< true, Real, Device, Index, ProblemSetter >::resolveGridGeometryType( const tnlParameterContainer& parameters,
                                                                                               const tnlList< tnlString >& parsedMeshType )
{
   tnlList< tnlString > parsedGeometryType;
   if( ! parseObjectType( parsedMeshType[ 5 ], parsedGeometryType ) )
   {
      cerr << "Unable to parse the geometry type " << parsedMeshType[ 5 ] << "." << endl;
      return false;
   }
   if( parsedGeometryType[ 0 ] == "tnlIdenticalGridGeometry" )
   {
      typedef tnlGrid< MeshDimensions, MeshRealType, Device, MeshIndexType, tnlIdenticalGridGeometry > MeshType;
      return ProblemSetter< MeshType, tnlSolverStarter >::template run< Real, Device, Index >( parameters );
   }
}

template< int Dimensions, typename MeshTypeResolver >
bool tnlMeshTypeResolverDimensionsSupportChecker< Dimensions, true, MeshTypeResolver >::checkDimensions( const tnlParameterContainer& parameters,
                                                                                                        const tnlList< tnlString >& parsedMeshType)
{
   return MeshTypeResolver::template resolveMeshRealType< Dimensions >( parameters, parsedMeshType );
};

template< int Dimensions, typename MeshTypeResolver >
bool tnlMeshTypeResolverDimensionsSupportChecker< Dimensions, false, MeshTypeResolver >::checkDimensions( const tnlParameterContainer& parameters,
                                                                                            const tnlList< tnlString >& parsedMeshType )
{
   cerr << Dimensions << " are not supported  by the solver." << endl;
   return false;
};



#endif /* TNLMESHTYPERESOLVER_IMPL_H_ */
