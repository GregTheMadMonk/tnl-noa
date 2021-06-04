/***************************************************************************
                          resolveMeshType.h  -  description
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

/**
 * This function does the following (in pseudo-code):
 *
 * \code
 * using Reader = [based on file type detection]
 * Reader reader;
 * using MeshType = [black magic]
 * MeshType mesh;
 * return functor( reader, mesh );
 * \endcode
 *
 * The functor should be a generic lambda expression with the following
 * signature (or an equivalent functor):
 *
 * \code
 * auto functor = [] ( auto& reader, auto&& mesh ) -> bool {};
 * \endcode
 */
template< typename ConfigTag,
          typename Device,
          typename Functor >
bool
resolveMeshType( Functor&& functor,
                 const std::string& fileName,
                 const std::string& fileFormat = "auto" );

/**
 * This function dues the same as \ref resolveMeshType, but also reuses the mesh
 * reader instance to load the mesh before passing it to the functor.
 * In pseudo-code:
 *
 * \code
 * using Reader = [based on file type detection]
 * Reader reader;
 * using MeshType = [black magic]
 * MeshType mesh;
 * if( ! reader.readMesh( mesh ) )
 *    return false;
 * return functor( reader, mesh );
 * \endcode
 */
template< typename ConfigTag,
          typename Device,
          typename Functor >
bool
resolveAndLoadMesh( Functor&& functor,
                    const std::string& fileName,
                    const std::string& fileFormat = "auto" );

/**
 * This function takes a file name and a mesh instance and attempts to load the
 * mesh from the file into the object.
 *
 * \remark The use of this function in combination with \ref resolveMeshType
 * should be avoided. Use \ref resolveAndLoadMesh instead to reuse the mesh
 * reader instance created in \ref resolveMeshType.
 */
template< typename Mesh >
bool
loadMesh( Mesh& mesh,
          const std::string& fileName,
          const std::string& fileFormat = "auto" );

template< typename MeshConfig >
bool
loadMesh( Mesh< MeshConfig, Devices::Cuda >& mesh,
          const std::string& fileName,
          const std::string& fileFormat = "auto" );

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/TypeResolver/resolveMeshType.hpp>
