/***************************************************************************
                          DistributedMesh.h  -  description
                             -------------------
    begin                : April 9, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <TNL/Containers/Array.h>
#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Meshes/DistributedMeshes/GlobalIndexStorage.h>

namespace TNL {
namespace Meshes {
namespace DistributedMeshes {

template< typename Mesh >
class DistributedMesh
: protected GlobalIndexStorage< Mesh, 0 >,
  protected GlobalIndexStorage< Mesh, Mesh::getMeshDimension() >
{
public:
   using MeshType           = Mesh;
   using Config             = typename Mesh::Config;
   using DeviceType         = typename Mesh::DeviceType;
   using GlobalIndexType    = typename Mesh::GlobalIndexType;
   using LocalIndexType     = typename Mesh::LocalIndexType;
   using PointType          = typename Mesh::PointType;
   using RealType           = typename PointType::RealType;
   using GlobalIndexArray   = typename Mesh::GlobalIndexArray;
   using CommunicatorType   = Communicators::MpiCommunicator;
   using CommunicationGroup = typename CommunicatorType::CommunicationGroup;
   using VTKTypesArrayType  = Containers::Array< std::uint8_t, Devices::Sequential, GlobalIndexType >;

   DistributedMesh() = default;

   DistributedMesh( MeshType&& localMesh )
   : localMesh( std::move(localMesh) )
   {}

   DistributedMesh( const DistributedMesh& ) = default;

   DistributedMesh( DistributedMesh&& ) = default;

   DistributedMesh& operator=( const DistributedMesh& ) = default;

   DistributedMesh& operator=( DistributedMesh&& ) = default;

   template< typename Mesh_ >
   DistributedMesh& operator=( const Mesh_& other )
   {
      localMesh = other.getLocalMesh();
      group = other.getCommunicationGroup();
      return *this;
   }

   /**
    * Common methods redirected to the local mesh
    */
   static constexpr int getMeshDimension()
   {
      return MeshType::getMeshDimension();
   }

   // types of common entities
   using Cell = typename MeshType::template EntityType< getMeshDimension() >;
   using Face = typename MeshType::template EntityType< getMeshDimension() - 1 >;
   using Vertex = typename MeshType::template EntityType< 0 >;

   static_assert( Mesh::Config::entityTagsStorage( typename Cell::EntityTopology{} ),
                  "DistributedMesh must store entity tags on cells" );
   static_assert( Mesh::Config::entityTagsStorage( typename Vertex::EntityTopology{} ),
                  "DistributedMesh must store entity tags on vertices" );


   /**
    * Methods specific to the distributed mesh
    */
   void setCommunicationGroup( CommunicationGroup group )
   {
      this->group = group;
   }

   CommunicationGroup getCommunicationGroup() const
   {
      return group;
   }

   const MeshType& getLocalMesh() const
   {
      return localMesh;
   }

   MeshType& getLocalMesh()
   {
      return localMesh;
   }

   void setGhostLevels( int levels )
   {
      ghostLevels = levels;
   }

   int getGhostLevels() const
   {
      return ghostLevels;
   }

   template< int Dimension >
   const GlobalIndexArray&
   getGlobalIndices() const
   {
      static_assert( Dimension == 0 || Dimension == MeshType::getMeshDimension(),
                     "Global index array for this dimension is not implemented yet." );
      return GlobalIndexStorage< MeshType, Dimension >::getGlobalIndices();
   }

   template< int Dimension >
   GlobalIndexArray&
   getGlobalIndices()
   {
      static_assert( Dimension == 0 || Dimension == MeshType::getMeshDimension(),
                     "Global index array for this dimension is not implemented yet." );
      return GlobalIndexStorage< MeshType, Dimension >::getGlobalIndices();
   }

   VTKTypesArrayType&
   vtkCellGhostTypes()
   {
      return vtkCellGhostTypesArray;
   }

   const VTKTypesArrayType&
   vtkCellGhostTypes() const
   {
      return vtkCellGhostTypesArray;
   }

   VTKTypesArrayType&
   vtkPointGhostTypes()
   {
      return vtkPointGhostTypesArray;
   }

   const VTKTypesArrayType&
   vtkPointGhostTypes() const
   {
      return vtkPointGhostTypesArray;
   }

   void
   printInfo( std::ostream& str ) const
   {
      const GlobalIndexType pointsCount = localMesh.template getEntitiesCount< 0 >();
      const GlobalIndexType cellsCount = localMesh.template getEntitiesCount< Mesh::getMeshDimension() >();

      // TODO: the mesh should explicitly store ghost counts (or offsets) - useful for efficient iteration
      GlobalIndexType ghostPoints = 0;
      for( GlobalIndexType i = 0; i < pointsCount; i++ )
         if( localMesh.template isGhostEntity< 0 >( i ) )
            ghostPoints++;
      GlobalIndexType ghostCells = 0;
      for( GlobalIndexType i = 0; i < cellsCount; i++ )
         if( localMesh.template isGhostEntity< Mesh::getMeshDimension() >( i ) )
            ghostCells++;

      CommunicatorType::Barrier();
      for( int i = 0; i < CommunicatorType::GetSize(); i++ ) {
         if( i == CommunicatorType::GetRank() ) {
            str << "MPI rank:\t" << CommunicatorType::GetRank() << "\n"
                << "\tMesh dimension:\t" << getMeshDimension() << "\n"
                << "\tCell topology:\t" << getType( typename Cell::EntityTopology{} ) << "\n"
                << "\tCells count:\t" << cellsCount << "\n"
                << "\tPoints count:\t" << pointsCount << "\n"
                << "\tGhost levels:\t" << getGhostLevels() << "\n"
                << "\tGhost cells:\t" << ghostCells << "\n"
                << "\tGhost points:\t" << ghostPoints << "\n";
            const GlobalIndexType globalPointIndices = getGlobalIndices< 0 >().getSize();
            const GlobalIndexType globalCellIndices = getGlobalIndices< Mesh::getMeshDimension() >().getSize();
            if( getGhostLevels() > 0 ) {
               if( globalPointIndices != pointsCount )
                  str << "ERROR: array of global point indices has wrong size: " << globalPointIndices << "\n";
               if( globalCellIndices != cellsCount )
                  str << "ERROR: array of global cell indices has wrong size: " << globalCellIndices << "\n";
               if( vtkPointGhostTypesArray.getSize() != pointsCount )
                  str << "ERROR: array of VTK point ghost types has wrong size: " << vtkPointGhostTypesArray.getSize() << "\n";
               if( vtkCellGhostTypesArray.getSize() != cellsCount )
                  str << "ERROR: array of VTK cell ghost types has wrong size: " << vtkCellGhostTypesArray.getSize() << "\n";
            }
            else {
               if( globalPointIndices > 0 )
                  str << "WARNING: mesh has 0 ghost levels, but array of global point indices has non-zero size: " << globalPointIndices << "\n";
               if( globalCellIndices > 0 )
                  str << "WARNING: mesh has 0 ghost levels, but array of global cell indices has non-zero size: " << globalCellIndices << "\n";
               if( vtkPointGhostTypesArray.getSize() > 0 )
                  str << "WARNING: mesh has 0 ghost levels, but array of VTK point ghost types has non-zero size: " << vtkPointGhostTypesArray.getSize() << "\n";
               if( vtkCellGhostTypesArray.getSize() > 0 )
                  str << "WARNING: mesh has 0 ghost levels, but array of VTK cell ghost types has non-zero size: " << vtkCellGhostTypesArray.getSize() << "\n";
            }
            str.flush();
         }
         CommunicatorType::Barrier();
      }
   }

protected:
   MeshType localMesh;
   CommunicationGroup group = CommunicatorType::NullGroup;
   int ghostLevels = 0;

   // vtkGhostType arrays for points and cells (cached for output into VTK formats)
   VTKTypesArrayType vtkPointGhostTypesArray, vtkCellGhostTypesArray;
};

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/DistributedMeshes/DistributedGrid.h>
