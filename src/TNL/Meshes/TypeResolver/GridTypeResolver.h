/***************************************************************************
                          GridTypeResolver.h  -  description
                             -------------------
    begin                : Nov 22, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>

#include <TNL/Meshes/BuildConfigTags.h>

namespace TNL {
namespace Meshes {

template< typename ConfigTag,
          typename Device >
class GridTypeResolver
{
public:

   template< typename Reader, typename Functor >
   static bool run( Reader& reader, Functor&& functor );

protected:
   template< typename Reader, typename Functor >
   struct detail
   {
      static bool resolveGridDimension( Reader& reader, Functor&& functor );

      // NOTE: We could disable the grids only by the GridTag, but doing the
      //       resolution for all subtypes is more flexible and also pretty
      //       good optimization of compilation times.

      // Overload for disabled grid dimensions
      template< int MeshDimension,
                typename = typename std::enable_if< ! BuildConfigTags::GridDimensionTag< ConfigTag, MeshDimension >::enabled >::type,
                typename = void >
      static bool resolveReal( Reader& reader, Functor&& functor );

      // Overload for enabled grid dimensions
      template< int MeshDimension,
                typename = typename std::enable_if< BuildConfigTags::GridDimensionTag< ConfigTag, MeshDimension >::enabled >::type >
      static bool resolveReal( Reader& reader, Functor&& functor );

      // Overload for disabled real types
      template< int MeshDimension,
                typename Real,
                typename = typename std::enable_if< ! BuildConfigTags::GridRealTag< ConfigTag, Real >::enabled >::type,
                typename = void >
      static bool resolveIndex( Reader& reader, Functor&& functor );

      // Overload for enabled real types
      template< int MeshDimension,
                typename Real,
                typename = typename std::enable_if< BuildConfigTags::GridRealTag< ConfigTag, Real >::enabled >::type >
      static bool resolveIndex( Reader& reader, Functor&& functor );

      // Overload for disabled index types
      template< int MeshDimension,
                typename Real,
                typename Index,
                typename = typename std::enable_if< ! BuildConfigTags::GridIndexTag< ConfigTag, Index >::enabled >::type,
                typename = void >
      static bool resolveGridType( Reader& reader, Functor&& functor );

      // Overload for enabled index types
      template< int MeshDimension,
                typename Real,
                typename Index,
                typename = typename std::enable_if< BuildConfigTags::GridIndexTag< ConfigTag, Index >::enabled >::type >
      static bool resolveGridType( Reader& reader, Functor&& functor );

      // Overload for disabled grid types
      template< typename GridType,
                typename = typename std::enable_if< ! BuildConfigTags::GridTag< ConfigTag, GridType >::enabled >::type,
                typename = void >
      static bool resolveTerminate( Reader& reader, Functor&& functor );

      // Overload for enabled grid types
      template< typename GridType,
                typename = typename std::enable_if< BuildConfigTags::GridTag< ConfigTag, GridType >::enabled >::type >
      static bool resolveTerminate( Reader& reader, Functor&& functor );
   };
};

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/TypeResolver/GridTypeResolver_impl.h>
