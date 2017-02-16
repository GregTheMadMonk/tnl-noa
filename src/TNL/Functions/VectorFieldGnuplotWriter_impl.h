/***************************************************************************
                          VectorFieldGnuplotWriter.h  -  description
                             -------------------
    begin                : Feb 16, 2017
    copyright            : (C) 2017 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Functions/VectorFieldGnuplotWriter.h>

namespace TNL {
namespace Functions {

template< typename VectorField >
bool
VectorFieldGnuplotWriter< VectorField >::
write( const VectorField& function,
       std::ostream& str )
{
   std::cerr << "Gnuplot writer for mesh functions defined on mesh type " << VectorField::MeshType::getType() << " is not (yet) implemented." << std::endl;
   return false;
}

/****
 * 1D grid, cells
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
bool
VectorFieldGnuplotWriter< VectorField< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 1, Real > >::
write( const VectorFieldType& function,
       std::ostream& str )
{
   const MeshType& mesh = function.getMesh();
   typename MeshType::Cell entity( mesh );
   for( entity.getCoordinates().x() = 0;
        entity.getCoordinates().x() < mesh.getDimensions().x();
        entity.getCoordinates().x() ++ )
   {
      entity.refresh();
      typename MeshType::VertexType v = entity.getCenter();
      str << v.x() << " "
          << function.getData().getElement( entity.getIndex() ) << std::endl;
   }
   return true;
}

/****
 * 1D grid, vertices
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
bool
VectorFieldGnuplotWriter< VectorField< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 0, Real > >::
write( const VectorFieldType& function,
       std::ostream& str )
{
   const MeshType& mesh = function.getMesh();
   typename MeshType::Vertex entity( mesh );
   for( entity.getCoordinates().x() = 0;
        entity.getCoordinates().x() <= mesh.getDimensions().x();
        entity.getCoordinates().x() ++ )
   {
      entity.refresh();
      typename MeshType::VertexType v = entity.getCenter();
      str << v.x() << " "
          << function.getData().getElement( entity.getIndex() ) << std::endl;
   }
   return true;
}


/****
 * 2D grid, cells
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
bool
VectorFieldGnuplotWriter< VectorField< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 2, Real > >::
write( const VectorFieldType& function,
       std::ostream& str )
{
   const MeshType& mesh = function.getMesh();
   typename MeshType::Cell entity( mesh );
   for( entity.getCoordinates().y() = 0;
        entity.getCoordinates().y() < mesh.getDimensions().y();
        entity.getCoordinates().y() ++ )
   {
      for( entity.getCoordinates().x() = 0;
           entity.getCoordinates().x() < mesh.getDimensions().x();
           entity.getCoordinates().x() ++ )
      {
         entity.refresh();
         typename MeshType::VertexType v = entity.getCenter();
         str << v.x() << " " << v.y() << " "
             << function.getData().getElement( entity.getIndex() ) << std::endl;
      }
      str << std::endl;
   }
   return true;
}

/****
 * 2D grid, faces
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
bool
VectorFieldGnuplotWriter< VectorField< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 1, Real > >::
write( const VectorFieldType& function,
       std::ostream& str )
{
   const MeshType& mesh = function.getMesh();
   typedef typename MeshType::Face EntityType;
   typedef typename EntityType::EntityOrientationType EntityOrientation;
   EntityType entity( mesh );

   entity.setOrientation( EntityOrientation( 1.0, 0.0 ) );
   for( entity.getCoordinates().y() = 0;
        entity.getCoordinates().y() < mesh.getDimensions().y();
        entity.getCoordinates().y() ++ )
   {
      for( entity.getCoordinates().x() = 0;
           entity.getCoordinates().x() <= mesh.getDimensions().x();
           entity.getCoordinates().x() ++ )
      {
         entity.refresh();
         typename MeshType::VertexType v = entity.getCenter();
         str << v.x() << " " << v.y() << " "
             << function.getData().getElement( entity.getIndex() ) << std::endl;
      }
      str << std::endl;
   }

   entity.setOrientation( EntityOrientation( 0.0, 1.0 ) );
         for( entity.getCoordinates().x() = 0;
           entity.getCoordinates().x() < mesh.getDimensions().x();
           entity.getCoordinates().x() ++ )

   {
            for( entity.getCoordinates().y() = 0;
        entity.getCoordinates().y() <= mesh.getDimensions().y();
        entity.getCoordinates().y() ++ )

      {
         entity.refresh();
         typename MeshType::VertexType v = entity.getCenter();
         str << v.x() << " " << v.y() << " "
             << function.getData().getElement( entity.getIndex() ) << std::endl;
      }
      str << std::endl;
   }
   return true;
}


/****
 * 2D grid, vertices
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
bool
VectorFieldGnuplotWriter< VectorField< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 0, Real > >::
write( const VectorFieldType& function,
       std::ostream& str )
{
   const MeshType& mesh = function.getMesh();
   typename MeshType::Vertex entity( mesh );
   for( entity.getCoordinates().y() = 0;
        entity.getCoordinates().y() <= mesh.getDimensions().y();
        entity.getCoordinates().y() ++ )
   {
      for( entity.getCoordinates().x() = 0;
           entity.getCoordinates().x() <= mesh.getDimensions().x();
           entity.getCoordinates().x() ++ )
      {
         entity.refresh();
         typename MeshType::VertexType v = entity.getCenter();
         str << v.x() << " " << v.y() << " "
             << function.getData().getElement( entity.getIndex() ) << std::endl;
      }
      str << std::endl;
   }
   return true;
}


/****
 * 3D grid, cells
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
bool
VectorFieldGnuplotWriter< VectorField< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 3, Real > >::
write( const VectorFieldType& function,
       std::ostream& str )
{
   const MeshType& mesh = function.getMesh();
   typename MeshType::Cell entity( mesh );
   for( entity.getCoordinates().z() = 0;
        entity.getCoordinates().z() < mesh.getDimensions().z();
        entity.getCoordinates().z() ++ )
      for( entity.getCoordinates().y() = 0;
           entity.getCoordinates().y() < mesh.getDimensions().y();
           entity.getCoordinates().y() ++ )
      {
         for( entity.getCoordinates().x() = 0;
              entity.getCoordinates().x() < mesh.getDimensions().x();
              entity.getCoordinates().x() ++ )
         {
            entity.refresh();
            typename MeshType::VertexType v = entity.getCenter();
            str << v.x() << " " << v.y() << " " << v.z() << " "
                << function.getData().getElement( entity.getIndex() ) << std::endl;
         }
         str << std::endl;
      }
   return true;
}

/****
 * 3D grid, faces
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
bool
VectorFieldGnuplotWriter< VectorField< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 2, Real > >::
write( const VectorFieldType& function,
       std::ostream& str )
{
   const MeshType& mesh = function.getMesh();
   typedef typename MeshType::Face EntityType;
   typedef typename EntityType::EntityOrientationType EntityOrientation;
   EntityType entity( mesh );

   entity.setOrientation( EntityOrientation( 1.0, 0.0, 0.0 ) );
   for( entity.getCoordinates().z() = 0;
        entity.getCoordinates().z() < mesh.getDimensions().z();
        entity.getCoordinates().z() ++ )
      for( entity.getCoordinates().y() = 0;
           entity.getCoordinates().y() < mesh.getDimensions().y();
           entity.getCoordinates().y() ++ )
      {
         for( entity.getCoordinates().x() = 0;
              entity.getCoordinates().x() <= mesh.getDimensions().x();
              entity.getCoordinates().x() ++ )
         {
            entity.refresh();
            typename MeshType::VertexType v = entity.getCenter();
            str << v.x() << " " << v.y() << " " << v.z() << " "
                << function.getData().getElement( entity.getIndex() ) << std::endl;
         }
         str << std::endl;
      }

   entity.setOrientation( EntityOrientation( 0.0, 1.0, 0.0 ) );
   for( entity.getCoordinates().z() = 0;
        entity.getCoordinates().z() < mesh.getDimensions().z();
        entity.getCoordinates().z() ++ )
      for( entity.getCoordinates().x() = 0;
           entity.getCoordinates().x() < mesh.getDimensions().x();
           entity.getCoordinates().x() ++ )
      {
         for( entity.getCoordinates().y() = 0;
              entity.getCoordinates().y() <= mesh.getDimensions().y();
              entity.getCoordinates().y() ++ )
         {
            entity.refresh();
            typename MeshType::VertexType v = entity.getCenter();
            str << v.x() << " " << v.y() << " " << v.z() << " "
                << function.getData().getElement( entity.getIndex() ) << std::endl;
         }
         str << std::endl;
      }

   entity.setOrientation( EntityOrientation( 0.0, 0.0, 1.0 ) );
   for( entity.getCoordinates().x() = 0;
        entity.getCoordinates().x() < mesh.getDimensions().x();
        entity.getCoordinates().x() ++ )
      for( entity.getCoordinates().y() = 0;
           entity.getCoordinates().y() <= mesh.getDimensions().y();
           entity.getCoordinates().y() ++ )
      {
         for( entity.getCoordinates().z() = 0;
              entity.getCoordinates().z() < mesh.getDimensions().z();
              entity.getCoordinates().z() ++ )
         {
            entity.refresh();
            typename MeshType::VertexType v = entity.getCenter();
            str << v.x() << " " << v.y() << " " << v.z() << " "
                << function.getData().getElement( entity.getIndex() ) << std::endl;
         }
         str << std::endl;
      }
   return true;
}


/****
 * 3D grid, vertices
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
bool
VectorFieldGnuplotWriter< VectorField< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 0, Real > >::
write( const VectorFieldType& function,
       std::ostream& str )
{
   const MeshType& mesh = function.getMesh();
   typename MeshType::Vertex entity( mesh );
   for( entity.getCoordinates().z() = 0;
        entity.getCoordinates().z() <= mesh.getDimensions().z();
        entity.getCoordinates().z() ++ )
      for( entity.getCoordinates().y() = 0;
           entity.getCoordinates().y() <= mesh.getDimensions().y();
           entity.getCoordinates().y() ++ )
      {
         for( entity.getCoordinates().x() = 0;
              entity.getCoordinates().x() <= mesh.getDimensions().x();
              entity.getCoordinates().x() ++ )
         {
            entity.refresh();
            typename MeshType::VertexType v = entity.getCenter();
            str << v.x() << " " << v.y() << " " << v.z() << " "
                << function.getData().getElement( entity.getIndex() ) << std::endl;
         }
         str << std::endl;
      }
   return true;
}

} // namespace Functions
} // namespace TNL

