/***************************************************************************
                          PVTIReader.h  -  description
                             -------------------
    begin                : Jun 26, 2021
    copyright            : (C) 2021 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <experimental/filesystem>

#include <TNL/MPI/Wrappers.h>
#include <TNL/MPI/Utils.h>
#include <TNL/Meshes/Readers/VTIReader.h>
#include <TNL/Meshes/MeshDetails/layers/EntityTags/Traits.h>

namespace TNL {
namespace Meshes {
namespace Readers {

class PVTIReader
: public XMLVTK
{
   std::string
   getSourcePath( std::string source )
   {
      namespace fs = std::experimental::filesystem;
      return fs::path(fileName).parent_path() / source;
   }

#ifdef HAVE_TINYXML2
   void readParallelImageData()
   {
      using namespace tinyxml2;

      // read the required attributes
      const std::string extent = getAttributeString( datasetElement, "WholeExtent" );
      const std::string origin = getAttributeString( datasetElement, "Origin" );
      const std::string spacing = getAttributeString( datasetElement, "Spacing" );

      // parse the extent
      {
         std::stringstream ss( extent );
         gridExtent.resize( 6, 0 );
         for( int i = 0; i < 6; i++ ) {
            ss >> gridExtent[i];
            // check conversion error
            if( ! ss.good() )
               throw MeshReaderError( "VTIReader", "invalid extent: not a number: " + extent );
         }
         // check remaining characters
         std::string suffix;
         ss >> std::ws >> suffix;
         if( ! suffix.empty() )
            throw MeshReaderError( "VTIReader", "invalid extent " + extent + ": trailing characters: " + suffix );
      }

      // parse the origin
      {
         std::stringstream ss( origin );
         gridOrigin.resize( 3 );
         for( int i = 0; i < 3; i++ ) {
            ss >> gridOrigin[i];
            // check conversion error
            if( ! ss.good() )
               throw MeshReaderError( "VTIReader", "invalid origin: not a number: " + origin );
         }
         // check remaining characters
         std::string suffix;
         ss >> std::ws >> suffix;
         if( ! suffix.empty() )
            throw MeshReaderError( "VTIReader", "invalid origin " + origin + ": trailing characters: " + suffix );
      }

      // parse the spacing
      {
         std::stringstream ss( spacing );
         gridSpacing.resize( 3 );
         for( int i = 0; i < 3; i++ ) {
            ss >> gridSpacing[i];
            // check conversion error
            if( ! ss.good() )
               throw MeshReaderError( "VTIReader", "invalid spacing: not a number: " + spacing );
            // check negative numbers
            if( gridSpacing[i] < 0 )
               throw MeshReaderError( "VTIReader", "invalid spacing: negative number: " + spacing );
         }
         // check remaining characters
         std::string suffix;
         ss >> std::ws >> suffix;
         if( ! suffix.empty() )
            throw MeshReaderError( "VTIReader", "invalid spacing " + spacing + ": trailing characters: " + suffix );
      }

      // determine the grid dimension
      int dim = 0;
      for( int i = 0; i < 3; i++ )
         if( gridSpacing[i] > 0 )
            dim++;
         else
            break;
      spaceDimension = meshDimension = dim;

      // populate cellShape (just for completeness, not necessary for GridTypeResolver)
      if( meshDimension == 1 )
         cellShape = VTK::EntityShape::Line;
      else if( meshDimension == 2 )
         cellShape = VTK::EntityShape::Pixel;
      else if( meshDimension == 3 )
         cellShape = VTK::EntityShape::Voxel;

      // RealType is not stored in the VTI file, so just set the default
      pointsType = "double";

      // set the index type
      connectivityType = headerType;


      // read GhostLevel attribute
      ghostLevels = getAttributeInteger( datasetElement, "GhostLevel" );
      // read MinCommonVertices attribute (TNL-specific, optional)
      minCommonVertices = getAttributeInteger( datasetElement, "MinCommonVertices", 0 );

      // read pieces info
      const XMLElement* piece = getChildSafe( datasetElement, "Piece" );
      while( piece ) {
         const std::string source = getAttributeString( piece, "Source" );
         if( source != "" ) {
            pieceSources.push_back( getSourcePath( source ) );
         }
         else
            throw MeshReaderError( "PVTIReader", "the Source attribute of a <Piece> element was found empty." );
         // find next
         piece = piece->NextSiblingElement( "Piece" );
      }
      if( pieceSources.size() == 0 )
         throw MeshReaderError( "PVTIReader", "the file does not contain any <Piece> element." );

      // check that the number of pieces matches the number of MPI ranks
      const int nproc = MPI::GetSize( communicator );
      if( (int) pieceSources.size() != nproc )
         throw MeshReaderError( "PVTIReader", "the number of subdomains does not match the number of MPI ranks ("
                                              + std::to_string(pieceSources.size()) + " vs " + std::to_string(nproc) + ")." );

      // read the local piece source
      const int rank = MPI::GetRank( communicator );
      localReader.setFileName( pieceSources[ rank ] );
      localReader.detectMesh();

      // check that local attributes are the same as global attributes
      if( getSpaceDimension() != localReader.getSpaceDimension() )
         throw MeshReaderError( "PVTIReader", "the space dimension of a subdomain does not match the global grid." );
      if( getMeshDimension() != localReader.getMeshDimension() )
         throw MeshReaderError( "PVTIReader", "the mesh dimension of a subdomain does not match the global grid." );
      if( getCellShape() != localReader.getCellShape() )
         throw MeshReaderError( "PVTIReader", "the cell shape of a subdomain does not match the global grid." );
      if( getRealType() != localReader.getRealType() )
         throw MeshReaderError( "PVTIReader", "the real type of a subdomain does not match the global grid." );
      if( getGlobalIndexType() != localReader.getGlobalIndexType() )
         throw MeshReaderError( "PVTIReader", "the global index type of a subdomain does not match the global grid." );

      // TODO: assert that all MPI ranks have the same attributes

      // TODO
//      if( ghostLevels > 0 ) {
//         // load the vtkGhostType arrays from PointData and CellData
//         pointTags = localReader.readPointData( VTK::ghostArrayName() );
//         cellTags = localReader.readCellData( VTK::ghostArrayName() );
//      }
   }
#endif

public:
   PVTIReader() = default;

   PVTIReader( const std::string& fileName, MPI_Comm communicator = MPI_COMM_WORLD )
   : XMLVTK( fileName ), communicator( communicator )
   {}

   virtual void detectMesh() override
   {
#ifdef HAVE_TINYXML2
      reset();
      try {
         openVTKFile();
      }
      catch( const MeshReaderError& ) {
         reset();
         throw;
      }

      // verify file type
      if( fileType == "PImageData" )
         readParallelImageData();
      else
         throw MeshReaderError( "PVTIReader", "the reader cannot read data of the type " + fileType + ". Use a different reader if possible." );

      // indicate success by setting the mesh type
      meshType = "Meshes::DistributedGrid";
#else
      throw_no_tinyxml();
#endif
   }

   template< typename MeshType >
   std::enable_if_t< isDistributedGrid< MeshType >::value >
   loadMesh( MeshType& mesh )
   {
      // check that detectMesh has been called
      if( meshType == "" )
         detectMesh();

      // check if we have a distributed grid
      if( meshType != "Meshes::DistributedGrid" )
         throw MeshReaderError( "MeshReader", "the file does not contain a distributed structured grid, it is " + meshType );

      // set the communicator
      mesh.setCommunicator( communicator );

      // TODO: set the domain decomposition
//      mesh.setDomainDecomposition( decomposition );

      // load the global grid (meshType must be faked before calling loadMesh)
      typename MeshType::GridType globalGrid;
      meshType = "Meshes::Grid";
      MeshReader::loadMesh( globalGrid );
      meshType = "Meshes::DistributedGrid";

      // set the global grid from the extent in the .pvti file
      // (this actually does the decomposition, i.e. computes the local size etc.)
      mesh.setGlobalGrid( globalGrid );

      // set ghost levels (this updates the lower and upper overlaps on the
      // distributed grid, as well as the localOrigin, localBegin and
      // localGridSize based on the overlaps)
      mesh.setGhostLevels( ghostLevels );
      // check MinCommonVertices
      // TODO
//      if( minCommonVertices > 0 && minCommonVertices != MeshType::Config::dualGraphMinCommonVertices )
//         std::cerr << "WARNING: the mesh was decomposed with different MinCommonVertices value than the value set in the mesh configuration "
//                      "(" << minCommonVertices << " vs " << MeshType::Config::dualGraphMinCommonVertices << ")." << std::endl;

      // load the local mesh and check with the subdomain
      typename MeshType::GridType localMesh;
      localReader.loadMesh( localMesh );
      if( localMesh != mesh.getLocalMesh() ) {
         std::stringstream msg;
         msg << "The grid from the " << MPI::GetRank( communicator ) << "-th subdomain .vti file does not match the local grid of the DistributedGrid."
             << "\n- Grid from the .vti file:\n" << localMesh
             << "\n- Local grid from the DistributedGrid:\n" << mesh.getLocalMesh();
         throw MeshReaderError( "PVTIReader", msg.str() );
      }

//      using Index = typename MeshType::IndexType;
//      const Index pointsCount = mesh.getLocalMesh().template getEntitiesCount< 0 >();
//      const Index cellsCount = mesh.getLocalMesh().template getEntitiesCount< MeshType::getMeshDimension() >();

/*    // TODO
      if( ghostLevels > 0 ) {
         // assign point ghost tags
         using mpark::get;
         const std::vector<std::uint8_t> pointTags = get< std::vector<std::uint8_t> >( this->pointTags );
         if( (Index) pointTags.size() != pointsCount )
            throw MeshReaderError( "PVTIReader", "the vtkGhostType array in PointData has wrong size: " + std::to_string(pointTags.size()) );
         mesh.vtkPointGhostTypes() = pointTags;
         for( Index i = 0; i < pointsCount; i++ )
            if( pointTags[ i ] & (std::uint8_t) VTK::PointGhostTypes::DUPLICATEPOINT )
               localMesh.template addEntityTag< 0 >( i, EntityTags::GhostEntity );

         // assign cell ghost tags
         using mpark::get;
         const std::vector<std::uint8_t> cellTags = get< std::vector<std::uint8_t> >( this->cellTags );
         if( (Index) cellTags.size() != cellsCount )
            throw MeshReaderError( "PVTIReader", "the vtkGhostType array in CellData has wrong size: " + std::to_string(cellTags.size()) );
         mesh.vtkCellGhostTypes() = cellTags;
         for( Index i = 0; i < cellsCount; i++ ) {
            if( cellTags[ i ] & (std::uint8_t) VTK::CellGhostTypes::DUPLICATECELL )
               localMesh.template addEntityTag< MeshType::getMeshDimension() >( i, EntityTags::GhostEntity );
         }

         // reset arrays since they are not needed anymore
         this->pointTags = this->cellTags = {};
      }
*/
   }

   template< typename MeshType >
   std::enable_if_t< ! isDistributedGrid< MeshType >::value >
   loadMesh( MeshType& mesh )
   {
      throw MeshReaderError( "MeshReader", "the PVTI reader cannot be used to load a distributed unstructured mesh." );
   }

   virtual VariantVector
   readPointData( std::string arrayName ) override
   {
      return localReader.readPointData( arrayName );
   }

   virtual VariantVector
   readCellData( std::string arrayName ) override
   {
      return localReader.readCellData( arrayName );
   }

   virtual void reset() override
   {
      resetBase();
      ghostLevels = 0;
      pieceSources = {};
      localReader.reset();
      pointTags = cellTags = {};
   }

protected:
   MPI_Comm communicator;

   int ghostLevels = 0;
   int minCommonVertices = 0;
   std::vector<std::string> pieceSources;

   VTIReader localReader;

   // additinal arrays we need to read from the localReader
   VariantVector pointTags, cellTags;
};

} // namespace Readers
} // namespace Meshes
} // namespace TNL
