/***************************************************************************
                          MeshResolver.h  -  description
                             -------------------
    begin                : Nov 22, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/Grid.h>

namespace TNL {
namespace Meshes {

/*
 * This function does the following (in pseudo-code):
 *
 *    using Reader = [based on file type detection]
 *    Reader reader;
 *    using MeshType = [black magic]
 *    MeshType mesh;
 *    return functor( reader, mesh );
 *
 * The functor should be a generic lambda expression with the following
 * signature (or an equivalent functor):
 *
 *    auto functor = [] ( auto& reader, auto&& mesh ) -> bool {};
 */
template< typename ConfigTag,
          typename Device,
          typename Functor >
bool resolveMeshType( const String& fileName, Functor&& functor );

/*
 * This function dues the same as `resolveMeshType`, but also reuses the mesh
 * reader instance to load the mesh before passing it to the functor.
 * In pseudo-code:
 *
 *    using Reader = [based on file type detection]
 *    Reader reader;
 *    using MeshType = [black magic]
 *    MeshType mesh;
 *    if( ! reader.readMesh( mesh ) )
 *       return false;
 *    return functor( reader, mesh );
 */
template< typename ConfigTag,
          typename Device,
          typename Functor >
bool resolveAndLoadMesh( const String& fileName, Functor&& functor );

/*
 * This function takes a file name and a mesh instance and attempts to load the
 * mesh from the file into the object.
 *
 * NOTE: The use of this function in combination with `resolveMeshType` should
 * be avoided. Use `resolveAndLoadMesh` instead to reuse the mesh reader
 * instance created in `resolveMeshType`.
 */
template< typename MeshConfig,
          typename Device >
bool
loadMesh( const String& fileName,
          Mesh< MeshConfig, Device >& mesh );

template< typename MeshConfig >
bool
loadMesh( const String& fileName,
          Mesh< MeshConfig, Devices::Cuda >& mesh );

template< int Dimension,
          typename Real,
          typename Device,
          typename Index >
bool
loadMesh( const String& fileName,
          Grid< Dimension, Real, Device, Index >& grid );

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/TypeResolver/TypeResolver_impl.h>
