/***************************************************************************
                          VTKWriter.h  -  description
                             -------------------
    begin                : Mar 04, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <limits>

#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/Mesh.h>

namespace TNL {
namespace Meshes {
namespace Writers {

namespace __impl {

template< typename Mesh, int EntityDimension > struct MeshEntitiesVTKWriter;
template< typename Mesh, int EntityDimension > struct MeshEntityTypesVTKWriter;

} // namespace __impl

template< typename Mesh >
class VTKWriter
{
   static_assert( Mesh::getMeshDimension() <= 3, "The VTK format supports only 1D, 2D and 3D meshes." );
   // TODO: check also world dimension when grids allow it
//   static_assert( Mesh::getWorldDimension() <= 3, "The VTK format supports only 1D, 2D and 3D meshes." );

   template< int EntityDimension >
   using EntitiesWriter = __impl::MeshEntitiesVTKWriter< Mesh, EntityDimension >;

   template< int EntityDimension >
   using EntityTypesWriter = __impl::MeshEntityTypesVTKWriter< Mesh, EntityDimension >;

public:
   using Index = typename Mesh::GlobalIndexType;

   static void writeAllEntities( const Mesh& mesh, std::ostream& str );

   template< int EntityDimension = Mesh::getMeshDimension() >
   static void writeEntities( const Mesh& mesh, std::ostream& str );

protected:
   static void writeHeader( const Mesh& mesh, std::ostream& str );

   static void writePoints( const Mesh& mesh, std::ostream& str );
};

} // namespace Writers
} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/Writers/VTKWriter_impl.h>
