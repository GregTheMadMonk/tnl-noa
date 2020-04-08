/***************************************************************************
                          GridTypeResolver_impl.h  -  description
                             -------------------
    begin                : Nov 22, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <utility>

#include <TNL/String.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/TypeResolver/GridTypeResolver.h>

namespace TNL {
namespace Meshes {

template< typename ConfigTag,
          typename Device >
   template< typename Reader,
             typename Functor >
bool
GridTypeResolver< ConfigTag, Device >::
run( Reader& reader, Functor&& functor )
{
   return detail< Reader, Functor >::resolveGridDimension( reader, std::forward<Functor>(functor) );
}

template< typename ConfigTag,
          typename Device >
   template< typename Reader,
             typename Functor >
bool
GridTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::
resolveGridDimension( Reader& reader, Functor&& functor )
{
   if( reader.getMeshDimension() == 1 )
      return resolveReal< 1 >( reader, std::forward<Functor>(functor) );
   if( reader.getMeshDimension() == 2 )
      return resolveReal< 2 >( reader, std::forward<Functor>(functor) );
   if( reader.getMeshDimension() == 3 )
      return resolveReal< 3 >( reader, std::forward<Functor>(functor) );
   std::cerr << "Unsupported mesh dimension: " << reader.getMeshDimension() << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device >
   template< typename Reader,
             typename Functor >
      template< int MeshDimension,
                typename, typename >
bool
GridTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::
resolveReal( Reader& reader, Functor&& functor )
{
   std::cerr << "The grid dimension " << MeshDimension << " is disabled in the build configuration." << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device >
   template< typename Reader,
             typename Functor >
      template< int MeshDimension,
                typename >
bool
GridTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::
resolveReal( Reader& reader, Functor&& functor )
{
   if( reader.getRealType() == "float" )
      return resolveIndex< MeshDimension, float >( reader, std::forward<Functor>(functor) );
   if( reader.getRealType() == "double" )
      return resolveIndex< MeshDimension, double >( reader, std::forward<Functor>(functor) );
   if( reader.getRealType() == "long-double" )
      return resolveIndex< MeshDimension, long double >( reader, std::forward<Functor>(functor) );
   std::cerr << "Unsupported real type: " << reader.getRealType() << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device >
   template< typename Reader,
             typename Functor >
      template< int MeshDimension,
                typename Real,
                typename, typename >
bool
GridTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::
resolveIndex( Reader& reader, Functor&& functor )
{
   std::cerr << "The grid real type " << getType< Real >() << " is disabled in the build configuration." << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device >
   template< typename Reader,
             typename Functor >
      template< int MeshDimension,
                typename Real,
                typename >
bool
GridTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::
resolveIndex( Reader& reader, Functor&& functor )
{
   if( reader.getGlobalIndexType() == "short int" )
      return resolveGridType< MeshDimension, Real, short int >( reader, std::forward<Functor>(functor) );
   if( reader.getGlobalIndexType() == "int" )
      return resolveGridType< MeshDimension, Real, int >( reader, std::forward<Functor>(functor) );
   if( reader.getGlobalIndexType() == "long int" )
      return resolveGridType< MeshDimension, Real, long int >( reader, std::forward<Functor>(functor) );
   std::cerr << "Unsupported index type: " << reader.getRealType() << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device >
   template< typename Reader,
             typename Functor >
      template< int MeshDimension,
                typename Real,
                typename Index,
                typename, typename >
bool
GridTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::
resolveGridType( Reader& reader, Functor&& functor )
{
   std::cerr << "The grid index type " << getType< Index >() << " is disabled in the build configuration." << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device >
   template< typename Reader,
             typename Functor >
      template< int MeshDimension,
                typename Real,
                typename Index,
                typename >
bool
GridTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::
resolveGridType( Reader& reader, Functor&& functor )
{
   using GridType = Meshes::Grid< MeshDimension, Real, Device, Index >;
   return resolveTerminate< GridType >( reader, std::forward<Functor>(functor) );
}

template< typename ConfigTag,
          typename Device >
   template< typename Reader,
             typename Functor >
      template< typename GridType,
                typename, typename >
bool
GridTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::
resolveTerminate( Reader& reader, Functor&& functor )
{
   std::cerr << "The mesh type " << TNL::getType< GridType >() << " is disabled in the build configuration." << std::endl;
   return false;
};

template< typename ConfigTag,
          typename Device >
   template< typename Reader,
             typename Functor >
      template< typename GridType,
                typename >
bool
GridTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::
resolveTerminate( Reader& reader, Functor&& functor )
{
   return std::forward<Functor>(functor)( reader, GridType{} );
};

} // namespace Meshes
} // namespace TNL
