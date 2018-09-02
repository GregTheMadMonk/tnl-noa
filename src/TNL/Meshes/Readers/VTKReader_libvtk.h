/***************************************************************************
                          VTKReader_libvtk.h  -  description
                             -------------------
    begin                : Nov 6, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <fstream>
#include <istream>
#include <vector>
#include <map>
#include <unordered_map>
#include <type_traits>

#include <TNL/Containers/StaticVector.h>
#include <TNL/Meshes/MeshBuilder.h>
#include <TNL/Meshes/Topologies/SubentityVertexMap.h>
#include <TNL/Meshes/Readers/EntityShape.h>

#ifdef HAVE_VTK
#include <vtkSmartPointer.h>
#include <vtkCell.h>
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridReader.h>
#endif

namespace TNL {
namespace Meshes {
namespace Readers {

// types used in VTK
using VTKRealType = double;
#ifdef HAVE_VTK
using VTKIndexType = vtkIdType;
#else
using VTKIndexType = int;
#endif
// wrapper type for physical coordinates to preserve my sanity
using VTKPointType = Containers::StaticVector< 3, VTKRealType >;

template< typename Index = int >
class VTKReader_libvtk
{
public:
   using IndexType = Index;

   bool
   detectMesh( const String& fileName )
   {
      resetAll();
      if( ! loadVTKFile( fileName ) ) {
         resetAll();
         return false;
      }
      reset();
      return true;
   }

   template< typename MeshType >
   bool
   readMesh( const String& fileName, MeshType& mesh )
   {
      static_assert( std::is_same< IndexType, typename MeshType::GlobalIndexType >::value, "VTKReader_libvtk::IndexType and MeshType::GlobalIndexType must be the same type." );

      resetAll();
      if( ! loadVTKFile( fileName ) ) {
         resetAll();
         return false;
      }

      // GOTCHA: The unary "+" is a workaround due to odr-use of undefined static data member. See:
      // https://stackoverflow.com/questions/39646958/constexpr-static-member-before-after-c17
      // https://stackoverflow.com/questions/272900/undefined-reference-to-static-class-member/272996#272996
      TNL_ASSERT_EQ( this->worldDimension, + MeshType::Config::worldDimension, "world dimensions do not match" );
      TNL_ASSERT_EQ( this->meshDimension, + MeshType::Config::meshDimension, "mesh dimensions do not match" );
      const int subvertices = Topologies::Subtopology< typename MeshType::Config::CellTopology, 0 >::count;
      TNL_ASSERT_EQ( this->verticesInEntities.at( this->meshDimension ), subvertices, "numbers of cell subvertices do not match" );

      using MeshBuilder = MeshBuilder< MeshType >;
      using GlobalIndexType = typename MeshType::GlobalIndexType;

      const GlobalIndexType numberOfPoints = this->pointsData.size();
      const GlobalIndexType numberOfCells = this->entityIdMappings.at( this->meshDimension ).size();

      MeshBuilder meshBuilder;
      meshBuilder.setPointsCount( numberOfPoints );
      meshBuilder.setCellsCount( numberOfCells );

      for( GlobalIndexType i = 0; i < numberOfPoints; i++ ) {
         typename MeshType::PointType p;
         for( int j = 0; j < p.size; j++ )
            p[ j ] = this->pointsData.at( i )[ j ];
         meshBuilder.setPoint( i, p );
      }

      const auto& cellIdMap = this->entityIdMappings.at( this->meshDimension );
      for( GlobalIndexType i = 0; i < numberOfCells; i++ ) {
         const VTKIndexType vtkCellIndex = cellIdMap.at( i );
         const auto& vtkCellSeeds = this->entitySeeds.at( vtkCellIndex );
         using CellSeedType = typename MeshBuilder::CellSeedType;
         TNL_ASSERT_EQ( CellSeedType::getCornersCount(), vtkCellSeeds.size(), "wrong number of subvertices" );
         CellSeedType& seed = meshBuilder.getCellSeed( i );
         for( int v = 0; v < CellSeedType::getCornersCount(); v++ ) {
            seed.setCornerId( v, vtkCellSeeds[ v ] );
         }
      }

      // drop all data from the VTK file since it's not needed anymore
      this->resetAll();

      return meshBuilder.build( mesh );
   }

   String
   getMeshType() const
   {
      // we can read only the UNSTRUCTURED_GRID dataset
      return "Meshes::Mesh";
   }

   int
   getWorldDimension() const
   {
      return this->worldDimension;
   }

   int
   getMeshDimension() const
   {
      return this->meshDimension;
   }

   EntityShape
   getCellShape() const
   {
      return this->entityTypes.at( this->meshDimension );
   }

//   int
//   getVerticesInCell() const
//   {
//      return this->verticesInEntities.at( this->getMeshDimension() );
//   }

//   EntityShape
//   getEntityType( int entityDimension ) const
//   {
//      return this->entityTypes.at( entityDimension );
//   }

   String
   getRealType() const
   {
      // TODO: how to extract it from the VTK object?
//      return "float";
      return "double";
   }

   String
   getGlobalIndexType() const
   {
      // not stored in the VTK file
      return "int";
   }

   String
   getLocalIndexType() const
   {
      // not stored in the VTK file
      return "short int";
   }

   String
   getIdType() const
   {
      // not stored in the VTK file
      return "int";
   }

protected:
   int worldDimension = 0;
   int meshDimension = 0;

   // maps vertex indices to physical coordinates
   std::unordered_map< VTKIndexType, VTKPointType > pointsData;

   // maps entity dimension to the number of vertices in the entity
   std::map< int, int > verticesInEntities;

   // type for mapping TNL entity indices to VTK cell indices
   using TNL2VTKindexmap = std::unordered_map< IndexType, VTKIndexType >;
   // maps dimension to maps of TNL entity IDs to corresponding VTK cell IDs
   std::unordered_map< int, TNL2VTKindexmap > entityIdMappings;

   // maps VTK cell indices to entity seeds (set of indices of subvertices)
   std::unordered_map< VTKIndexType, std::vector< VTKIndexType > > entitySeeds;

   // maps dimension to VTK type of the entity with given dimension
   std::unordered_map< int, EntityShape > entityTypes;

   void reset()
   {
      pointsData.clear();
      verticesInEntities.clear();
      entityIdMappings.clear();
      entitySeeds.clear();
   }

   void resetAll()
   {
      reset();
      worldDimension = 0;
      meshDimension = 0;
      entityTypes.clear();
   }

   bool loadVTKFile( const String& fileName )
   {
#ifdef HAVE_VTK
      vtkSmartPointer< vtkUnstructuredGridReader > vtkReader
         = vtkSmartPointer<vtkUnstructuredGridReader>::New();
      vtkReader->SetFileName( fileName.getString() );
      vtkReader->Update();

      if( ! vtkReader->IsFileUnstructuredGrid() ) {
         std::cerr << "The file '" << fileName << "' is not a VTK Legacy file with an UNSTRUCTURED_GRID dataset type." << std::endl;
         return false;
      }

      auto& vtkMesh = *vtkReader->GetOutput();

      /* To determine the mesh type, we need:
       * - world dimension (default 1D, if some 2nd coordinate is non-zero -> 2D, some 3rd coordinate non-zero -> 3D)
       * - mesh dimension (highest dimension of the entities present in mesh)
       * - check that it is homogeneous - all entities of the same dimension should have the same number of vertices and type
       * - types of entities for each dimension
       *
       * To initialize the mesh, we need:
       * - points data (vertex indices and world coordinates)
       * - cell seeds (indices of vertices composing each cell)
       */
      const VTKIndexType numberOfPoints = vtkMesh.GetNumberOfPoints();
      const VTKIndexType numberOfCells = vtkMesh.GetNumberOfCells();

      if( numberOfPoints == 0 ) {
         std::cerr << "There are no points data in the file '" << fileName << "'." << std::endl;
         return false;
      }

      if( numberOfCells == 0 ) {
         std::cerr << "There are no cells data in the file '" << fileName << "'." << std::endl;
         return false;
      }

      for( VTKIndexType i = 0; i < numberOfPoints; i++ ) {
         VTKRealType* p = vtkMesh.GetPoint( i );

         // get world dimension
         for( int j = 0; j < 3; j++ )
            if( p[ j ] != 0.0 )
               this->worldDimension = std::max( this->worldDimension, j + 1 );

         // copy points data
         this->pointsData[ i ] = VTKPointType( p[ 0 ], p[ 1 ], p[ 2 ] );
      }

      for( VTKIndexType i = 0; i < numberOfCells; i++ ) {
         vtkCell* cell = vtkMesh.GetCell( i );
         const int dimension = cell->GetCellDimension();
         const int points = cell->GetNumberOfPoints();
         const EntityShape type = (EntityShape) cell->GetCellType();

         // number of vertices in entities
         if( this->verticesInEntities.find( dimension ) == this->verticesInEntities.cend() )
            this->verticesInEntities[ dimension ] = points;
         else if( this->verticesInEntities[ dimension ] != points ) {
            std::cerr << "Mixed unstructured meshes are not supported. There are elements of dimension " << dimension
                      << " with " << this->verticesInEntities[ dimension ] << " vertices and with " << points
                      << " vertices. The number of vertices per entity must be constant." << std::endl;
            this->reset();
            return false;
         }

         // entity types
         if( this->entityTypes.find( dimension ) == this->entityTypes.cend() )
            this->entityTypes.emplace( std::make_pair( dimension, type ) );
         else if( this->entityTypes[ dimension ] != type ) {
            std::cerr << "Mixed unstructured meshes are not supported. There are elements of dimension " << dimension
                      << " with type " << this->entityTypes[ dimension ] << " and " << type
                      << ". The type of all entities with the same dimension must be the same." << std::endl;
            this->reset();
            return false;
         }

         // mapping between TNL and VTK indices
         auto& map = this->entityIdMappings[ dimension ];
         map[ map.size() ] = i;

         // copy seed
         auto& seed = this->entitySeeds[ i ];
         for( int j = 0; j < points; j++ )
            seed.push_back( cell->GetPointId( j ) );
      }

      // std::map is sorted, so the last key is the maximum
      this->meshDimension = this->verticesInEntities.rbegin()->first;

      if( this->meshDimension > this->worldDimension ) {
         std::cerr << "Invalid mesh: world dimension is " << this->worldDimension
                   << ", but mesh dimension is " << this->meshDimension << "." << std::endl;
         this->reset();
         return false;
      }

      return true;
#else
      std::cerr << "The VTKReader_libvtk needs to be compiled with the VTK library." << std::endl;
      return false;
#endif
   }
};

} // namespace Readers
} // namespace Meshes
} // namespace TNL
