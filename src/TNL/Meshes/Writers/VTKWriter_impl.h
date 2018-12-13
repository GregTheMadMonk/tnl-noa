/***************************************************************************
                          VTKWriter.h  -  description
                             -------------------
    begin                : Mar 04, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>

#include <TNL/Meshes/Writers/VTKWriter.h>
#include <TNL/Meshes/Readers/EntityShape.h>

namespace TNL {
namespace Meshes {
namespace Writers {

namespace __impl {

template< typename T, typename R = void >
struct enable_if_type
{
   using type = R;
};

template< typename T, typename Enable = void >
struct has_entity_topology : std::false_type {};

template< typename T >
struct has_entity_topology< T, typename enable_if_type< typename T::EntityTopology >::type >
: std::true_type
{};


template< typename Entity,
          bool _is_mesh_entity = has_entity_topology< Entity >::value >
struct VerticesPerEntity
{
   static constexpr int count = Entity::getVerticesCount();
};

template< typename MeshConfig, typename Device >
struct VerticesPerEntity< MeshEntity< MeshConfig, Device, Topologies::Vertex >, true >
{
   static constexpr int count = 1;
};

template< typename GridEntity >
struct VerticesPerEntity< GridEntity, false >
{
private:
   static constexpr int dim = GridEntity::getEntityDimension();
   static_assert( dim >= 0 && dim <= 3, "unexpected dimension of the grid entity" );

public:
   static constexpr int count =
      (dim == 0) ? 1 :
      (dim == 1) ? 2 :
      (dim == 2) ? 4 :
                   8;
};


template< typename GridEntity >
struct GridEntityShape
{
private:
   static constexpr int dim = GridEntity::getEntityDimension();
   static_assert( dim >= 0 && dim <= 3, "unexpected dimension of the grid entity" );

public:
   static constexpr Readers::EntityShape shape =
      (dim == 0) ? Readers::EntityShape::Vertex :
      (dim == 1) ? Readers::EntityShape::Line :
      (dim == 2) ? Readers::EntityShape::Pixel :
                   Readers::EntityShape::Voxel;
};


template< typename Mesh >
typename Mesh::GlobalIndexType
getAllMeshEntitiesCount( const Mesh& mesh, DimensionTag< 0 > )
{
   using EntityType = typename Mesh::template EntityType< 0 >;
   return mesh.template getEntitiesCount< EntityType >();
}

// TODO: specialization for disabled entities
template< typename Mesh,
          typename DimensionTag = Meshes::DimensionTag< Mesh::getMeshDimension() > >
typename Mesh::GlobalIndexType
getAllMeshEntitiesCount( const Mesh& mesh, DimensionTag = DimensionTag() )
{
   using EntityType = typename Mesh::template EntityType< DimensionTag::value >;
   return mesh.template getEntitiesCount< EntityType >() +
          getAllMeshEntitiesCount( mesh, typename DimensionTag::Decrement() );
}


template< typename Mesh >
typename Mesh::GlobalIndexType
getCellsListSize( const Mesh& mesh, DimensionTag< 0 > )
{
   using EntityType = typename Mesh::template EntityType< 0 >;
   return mesh.template getEntitiesCount< EntityType >() * 2;
}

// TODO: specialization for disabled entities
template< typename Mesh,
          typename DimensionTag = Meshes::DimensionTag< Mesh::getMeshDimension() > >
typename Mesh::GlobalIndexType
getCellsListSize( const Mesh& mesh, DimensionTag = DimensionTag() )
{
   using EntityType = typename Mesh::template EntityType< DimensionTag::value >;
   const auto verticesPerEntity = VerticesPerEntity< EntityType >::count;
   return ( mesh.template getEntitiesCount< EntityType >() * ( verticesPerEntity + 1 ) ) +
          getCellsListSize( mesh, typename DimensionTag::Decrement() );
}


// TODO: specialization for disabled entities
// Unstructured meshes, entities
template< typename Mesh, int EntityDimension >
struct MeshEntitiesVTKWriter
{
   static void exec( const Mesh& mesh, std::ostream& str )
   {
      using EntityType = typename Mesh::template EntityType< EntityDimension >;
      using Index = typename Mesh::GlobalIndexType;

      const Index entitiesCount = mesh.template getEntitiesCount< EntityType >();
      const Index verticesPerEntity = VerticesPerEntity< EntityType >::count;;
      for( Index i = 0; i < entitiesCount; i++ ) {
         const auto& entity = mesh.template getEntity< EntityType >( i );
         str << verticesPerEntity;
         for( Index j = 0; j < verticesPerEntity; j++ )
            str << " " << entity.template getSubentityIndex< 0 >( j );
         str << "\n";
      }
   }
};

// Unstructured meshes, vertices
template< typename Mesh >
struct MeshEntitiesVTKWriter< Mesh, 0 >
{
   static void exec( const Mesh& mesh, std::ostream& str )
   {
      using EntityType = typename Mesh::template EntityType< 0 >;
      using Index = typename Mesh::GlobalIndexType;

      const Index entitiesCount = mesh.template getEntitiesCount< EntityType >();
      const Index verticesPerEntity = 1;
      for( Index i = 0; i < entitiesCount; i++ ) {
         str << verticesPerEntity << " " << i << "\n";
      }
   }
};

// 1D grids, cells
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTKWriter< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 1 >
{
   using MeshType = Meshes::Grid< 1, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str )
   {
      for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
         str << "2 " << i << " " << i+1 << "\n";
   }
};

// 1D grids, vertices
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTKWriter< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 0 >
{
   using MeshType = Meshes::Grid< 1, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str )
   {
      for( MeshIndex i = 0; i < mesh.getDimensions().x() + 1; i++ )
         str << "1 " << i << "\n";
   }
};

// 2D grids, cells
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTKWriter< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 2 >
{
   using MeshType = Meshes::Grid< 2, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str )
   {
      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
         for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
            str << "4 " << j * ( mesh.getDimensions().x() + 1 ) + i << " "
                        << j * ( mesh.getDimensions().x() + 1 ) + i + 1 << " "
                        << (j+1) * ( mesh.getDimensions().x() + 1 ) + i << " "
                        << (j+1) * ( mesh.getDimensions().x() + 1 ) + i + 1 << "\n";
   }
};

// 2D grids, faces
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTKWriter< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 1 >
{
   using MeshType = Meshes::Grid< 2, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str )
   {
      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
         for( MeshIndex i = 0; i < ( mesh.getDimensions().x() + 1 ); i++ )
            str << "2 " << j * ( mesh.getDimensions().x() + 1 ) + i << " "
                        << (j+1) * ( mesh.getDimensions().x() + 1 ) + i << "\n";

      for( MeshIndex j = 0; j < (mesh.getDimensions().y()+1); j++ )
         for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
            str << "2 " << j * ( mesh.getDimensions().x() + 1 ) + i << " "
                        << j * ( mesh.getDimensions().x() + 1 ) + i + 1 << "\n";
   }
};

// 2D grids, vertices
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTKWriter< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 0 >
{
   using MeshType = Meshes::Grid< 2, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str )
   {
      for( MeshIndex j = 0; j < ( mesh.getDimensions().y() + 1 ); j++ )
         for( MeshIndex i = 0; i < ( mesh.getDimensions().x() + 1 ); i++ )
            str << "1 " << j * mesh.getDimensions().x() + i << "\n";
   }
};

// 3D grids, cells
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTKWriter< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 3 >
{
   using MeshType = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str )
   {
      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
               str << "8 " << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i << " "
                           << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 << " "
                           << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i << " "
                           << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i + 1 << " "
                           << (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i << " "
                           << (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 << " "
                           << (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i << " "
                           << (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i + 1 << "\n";
   }
};

// 3D grids, faces
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTKWriter< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 2 >
{
   using MeshType = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str )
   {
      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i <= mesh.getDimensions().x(); i++ )
               str << "4 " << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i << " "
                           << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i << " "
                           << (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i << " "
                           << (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i << "\n";

      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j <= mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
               str << "4 " << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i << " "
                           << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 << " "
                           << (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i << " "
                           << (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 << "\n";

      for( MeshIndex k = 0; k <= mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
               str << "4 " << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i << " "
                           << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 << " "
                           << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i << " "
                           << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i + 1 << "\n";
   }
};

// 3D grids, edges
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTKWriter< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 1 >
{
   using MeshType = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str )
   {
      for( MeshIndex k = 0; k <= mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j <= mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
               str << "2 " << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i << " "
                           << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 << "\n";

      for( MeshIndex k = 0; k <= mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i <= mesh.getDimensions().x(); i++ )
               str << "2 " << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i << " "
                           << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i << "\n";

      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j <= mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i <= mesh.getDimensions().x(); i++ )
               str << "2 " << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i << " "
                           << (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i << "\n";
   }
};

// 3D grids, vertices
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTKWriter< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 0 >
{
   using MeshType = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str )
   {
      for( MeshIndex k = 0; k < ( mesh.getDimensions().z() + 1 ); k++ )
         for( MeshIndex j = 0; j < ( mesh.getDimensions().y() + 1 ); j++ )
            for( MeshIndex i = 0; i < ( mesh.getDimensions().x() + 1 ); i++ )
               str << "1 " << k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i  << "\n";
   }
};


// TODO: specialization for disabled entities
template< typename Mesh, int EntityDimension >
struct MeshEntityTypesVTKWriter
{
   static void exec( const Mesh& mesh, std::ostream& str )
   {
      using EntityType = typename Mesh::template EntityType< EntityDimension >;
      using Index = typename Mesh::GlobalIndexType;

      const Index entitiesCount = mesh.template getEntitiesCount< EntityType >();
      for( Index i = 0; i < entitiesCount; i++ ) {
         const int type = (int) Meshes::Readers::TopologyToEntityShape< typename EntityType::EntityTopology >::shape;
         str << type << "\n";
      }
   }
};

template< int Dimension,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          int EntityDimension >
struct MeshEntityTypesVTKWriter< Grid< Dimension, MeshReal, Device, MeshIndex >, EntityDimension >
{
   using MeshType = Grid< Dimension, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str )
   {
      using EntityType = typename MeshType::template EntityType< EntityDimension >;

      const MeshIndex entitiesCount = mesh.template getEntitiesCount< EntityType >();
      for( MeshIndex i = 0; i < entitiesCount; i++ ) {
         const int type = (int) __impl::GridEntityShape< EntityType >::shape;
         str << type << "\n";
      }
   }
};

} // namespace __impl

template< typename Mesh >
void
VTKWriter< Mesh >::writeAllEntities( const Mesh& mesh, std::ostream& str )
{
   writeHeader( mesh, str );
   writePoints( mesh, str );

   const Index allEntitiesCount = __impl::getAllMeshEntitiesCount( mesh );
   const Index cellsListSize = __impl::getCellsListSize( mesh );

   str << std::endl << "CELLS " << allEntitiesCount << " " << cellsListSize << std::endl;
   StaticFor< int, 0, Mesh::getMeshDimension() + 1, EntitiesWriter >::exec( mesh, str );

   str << std::endl << "CELL_TYPES " << allEntitiesCount << std::endl;
   StaticFor< int, 0, Mesh::getMeshDimension() + 1, EntityTypesWriter >::exec( mesh, str );
}

template< typename Mesh >
   template< int EntityDimension >
void
VTKWriter< Mesh >::writeEntities( const Mesh& mesh, std::ostream& str )
{
   writeHeader( mesh, str );
   writePoints( mesh, str );

   using EntityType = typename Mesh::template EntityType< EntityDimension >;
   const Index entitiesCount = mesh.template getEntitiesCount< EntityType >();
   const Index verticesPerEntity = __impl::VerticesPerEntity< EntityType >::count;
   const Index cellsListSize = entitiesCount * ( verticesPerEntity + 1 );

   str << std::endl << "CELLS " << entitiesCount << " " << cellsListSize << std::endl;
   EntitiesWriter< EntityDimension >::exec( mesh, str );

   str << std::endl << "CELL_TYPES " << entitiesCount << std::endl;
   EntityTypesWriter< EntityDimension >::exec( mesh, str );
}

template< typename Mesh >
void
VTKWriter< Mesh >::writeHeader( const Mesh& mesh, std::ostream& str )
{
    str << "# vtk DataFile Version 2.0\n"
        << "TNL DATA\n"
        << "ASCII\n"
        << "DATASET UNSTRUCTURED_GRID\n";
}

template< typename Mesh >
void
VTKWriter< Mesh >::writePoints( const Mesh& mesh, std::ostream& str )
{
   const Index verticesCount = mesh.template getEntitiesCount< typename Mesh::Vertex >();

   str << "POINTS " << verticesCount << " " << getType< typename Mesh::RealType >() << std::endl;
   str.precision( std::numeric_limits< typename Mesh::RealType >::digits10 );

   for( Index i = 0; i < verticesCount; i++ ) {
      const auto& vertex = mesh.template getEntity< typename Mesh::Vertex >( i );
      const auto& point = vertex.getPoint();
      for( Index j = 0; j < point.size; j++ ) {
         str << point[ j ];
         if( j < point.size - 1 )
            str << " ";
      }
      // VTK needs zeros for unused dimensions
      for( Index j = 0; j < 3 - point.size; j++ )
         str << " 0";
      str << "\n";
   }
}

} // namespace Writers
} // namespace Meshes
} // namespace TNL
