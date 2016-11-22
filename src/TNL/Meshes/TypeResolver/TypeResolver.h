/***************************************************************************
                          MeshResolver.h  -  description
                             -------------------
    begin                : Nov 22, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/String.h>

namespace TNL {
namespace Meshes {

/*
 * This function does the following (in pseudo-code):
 *
 *    using MeshType = [black magic]
 *    return ProblemSetter< MeshType >::run( problemSetterArgs... );
 */
template< typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
bool resolveMeshType( const String& fileName,
                      ProblemSetterArgs&&... problemSetterArgs );

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/TypeResolver/TypeResolver_impl.h>
