/***************************************************************************
                          VTKReader.h  -  description
                             -------------------
    begin                : Feb 25, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <fstream>
#include <sstream>
#include <map>
#include <vector>

#include <TNL/Meshes/MeshBuilder.h>
#include <TNL/Meshes/Readers/VTKEntityType.h>

namespace TNL {
namespace Meshes {
namespace Readers {

class VTKReader
{
public:
   bool detectMesh( const String& fileName )
   {
      this->reset();
      this->fileName = fileName;

      std::fstream inputFile( fileName.getString() );
      if( ! inputFile ) {
         std::cerr << "Failed to open the file " << fileName << "." << std::endl;
         return false;
      }

      if( ! parseHeader( inputFile ) )
         return false;

      if( dataset != "UNSTRUCTURED_GRID" ) {
         std::cerr << "VTKReader: the dataset '" << dataset << "' is not supported." << std::endl;
         return false;
      }

      // TODO: implement binary parsing
      if( dataType == "BINARY" ) {
         std::cerr << "VTKReader: parsing of BINARY data is not implemented yet." << std::endl;
         return false;
      }

      std::string line, aux;
      std::istringstream iss;

      // parse points section
      if( ! findSection( inputFile, "POINTS" ) ) {
         std::cerr << "VTKReader: unable to find the POINTS section, the file may be invalid or corrupted." << std::endl;
         return false;
      }
      getline( inputFile, line );
      iss.clear();
      iss.str( line );
      iss >> aux;
      long int numberOfVertices;
      iss >> numberOfVertices;
      iss >> realType;

      // read points
      long int verticesRead = 0;
      worldDimension = 0;
      while( verticesRead < numberOfVertices ) {
         if( ! inputFile ) {
            std::cerr << "VTKReader: unable to read enough vertices, the file may be invalid or corrupted." << std::endl;
            return false;
         }
         getline( inputFile, line );

         // read the coordinates and compute the world dimension
         iss.clear();
         iss.str( line );
         for( int i = 0; i < 3; i++ ) {
            double aux;
            iss >> aux;
            if( ! iss ) {
               std::cerr << "VTKReader: unable to read " << i << "th component of the vertex number " << verticesRead << "." << std::endl;
               return false;
            }
            if( aux != 0.0 )
               worldDimension = std::max( worldDimension, i + 1 );
         }

         verticesRead++;
      }

      // skip to the CELL_TYPES section
      if( ! findSection( inputFile, "CELL_TYPES" ) ) {
         std::cerr << "VTKReader: unable to find the CELL_TYPES section, the file may be invalid or corrupted." << std::endl;
         return false;
      }
      getline( inputFile, line );
      iss.clear();
      iss.str( line );
      iss >> aux;
      long int numberOfEntities;
      iss >> numberOfEntities;

      // read entity types
      long int entitiesRead = 0;
      std::map< int, VTKEntityType > entityTypes;
      while( entitiesRead < numberOfEntities ) {
         if( ! inputFile ) {
            std::cerr << "VTKReader: unable to read enough entity types, the file may be invalid or corrupted." << std::endl;
            return false;
         }
         getline( inputFile, line );

         // get entity type
         int typeId;
         iss.clear();
         iss.str( line );
         iss >> typeId;
         const VTKEntityType type = (VTKEntityType) typeId;
         const int dimension = getVTKEntityDimension( type );

         // check entity type
         if( entityTypes.find( dimension ) == entityTypes.cend() )
            entityTypes.emplace( std::make_pair( dimension, type ) );
         else if( entityTypes[ dimension ] != type ) {
            std::cerr << "Mixed unstructured meshes are not supported. There are elements of dimension " << dimension
                      << " with type " << entityTypes[ dimension ] << " and " << type << ". "
                      << "The type of all entities with the same dimension must be the same." << std::endl;
            this->reset();
            return false;
         }

         entitiesRead++;
      }

      // set meshDimension and cellVTKType
      meshDimension = 0;
      for( auto it : entityTypes )
         if( it.first > meshDimension ) {
            meshDimension = it.first;
            cellVTKType = it.second;
         }

      return true;
   }

   template< typename MeshType >
   bool readMesh( const String& fileName, MeshType& mesh )
   {
      using MeshBuilder = MeshBuilder< MeshType >;
      using IndexType = typename MeshType::GlobalIndexType;
      using PointType = typename MeshType::PointType;
      using CellSeedType = typename MeshBuilder::CellSeedType;

      const VTKEntityType cellType = TopologyToVTKMap< typename MeshType::template EntityTraits< MeshType::getMeshDimension() >::EntityTopology >::type;
      MeshBuilder meshBuilder;

      std::fstream inputFile( fileName.getString() );
      if( ! inputFile ) {
         std::cerr << "Failed to open the file " << fileName << "." << std::endl;
         return false;
      }

      if( ! parseHeader( inputFile ) )
         return false;
      const auto positionAfterHeading = inputFile.tellg();

      if( dataset != "UNSTRUCTURED_GRID" ) {
         std::cerr << "VTKReader: the dataset '" << dataset << "' is not supported." << std::endl;
         return false;
      }

      // TODO: implement binary parsing
      if( dataType == "BINARY" ) {
         std::cerr << "VTKReader: parsing of BINARY data is not implemented yet." << std::endl;
         return false;
      }

      std::string line, aux;
      std::istringstream iss;

      // parse the points section
      if( ! findSection( inputFile, "POINTS" ) ) {
         std::cerr << "VTKReader: unable to find the POINTS section, the file may be invalid or corrupted." << std::endl;
         return false;
      }
      getline( inputFile, line );
      iss.clear();
      iss.str( line );
      iss >> aux;
      IndexType numberOfVertices;
      iss >> numberOfVertices;

      // allocate vertices
      if( ! meshBuilder.setPointsCount( numberOfVertices ) ) {
         std::cerr << "Failed to allocate memory for " << numberOfVertices << " vertices." << std::endl;
         return false;
      }

      // read points
      for( IndexType vertexIndex = 0; vertexIndex < numberOfVertices; vertexIndex++ ) {
         if( ! inputFile ) {
            std::cerr << "VTKReader: unable to read enough vertices, the file may be invalid or corrupted." << std::endl;
            return false;
         }
         getline( inputFile, line );

         // read the coordinates
         iss.clear();
         iss.str( line );
         PointType p;
         for( int i = 0; i < p.size; i++ ) {
            iss >> p[ i ];
            if( ! iss ) {
               std::cerr << "VTKReader: unable to read " << i << "th component of the vertex number " << vertexIndex << "." << std::endl;
               return false;
            }
         }
         meshBuilder.setPoint( vertexIndex, p );
      }

      // find to the CELL_TYPES section
      if( ! findSection( inputFile, "CELL_TYPES", positionAfterHeading ) ) {
         std::cerr << "VTKReader: unable to find the CELL_TYPES section, the file may be invalid or corrupted." << std::endl;
         return false;
      }
      getline( inputFile, line );
      iss.clear();
      iss.str( line );
      iss >> aux;
      IndexType numberOfEntities;
      iss >> numberOfEntities;

      // read entity types, count cells
      std::vector< VTKEntityType > entityTypes;
      entityTypes.resize( numberOfEntities );
      IndexType numberOfCells = 0;
      for( IndexType entityIndex = 0; entityIndex < numberOfEntities; entityIndex++ ) {
         if( ! inputFile ) {
            std::cerr << "VTKReader: unable to read enough entity types, the file may be invalid or corrupted." << std::endl;
            return false;
         }
         getline( inputFile, line );

         // get entity type
         int typeId;
         iss.clear();
         iss.str( line );
         iss >> typeId;
         entityTypes[ entityIndex ] = (VTKEntityType) typeId;
         const int dimension = getVTKEntityDimension( entityTypes[ entityIndex ] );
         if( dimension == MeshType::getMeshDimension() )
            numberOfCells++;
      }

      if( ! meshBuilder.setCellsCount( numberOfCells ) ) {
         std::cerr << "Failed to allocate memory for " << numberOfCells << " cells." << std::endl;
         return false;
      }

      // find to the CELLS section
      if( ! findSection( inputFile, "CELLS", positionAfterHeading ) ) {
         std::cerr << "VTKReader: unable to find the CELLS section, the file may be invalid or corrupted." << std::endl;
         return false;
      }
      getline( inputFile, line );

      // read cells
      IndexType cellIndex = 0;
      for( IndexType entityIndex = 0; entityIndex < numberOfEntities; entityIndex++ ) {
         if( ! inputFile ) {
            std::cerr << "VTKReader: unable to read enough entities, the file may be invalid or corrupted." << std::endl;
            return false;
         }
         getline( inputFile, line );

         if( entityTypes[ entityIndex ] == cellType ) {
            iss.clear();
            iss.str( line );
            int vid;
            iss >> vid;  // ignore number of subvertices
            CellSeedType& seed = meshBuilder.getCellSeed( cellIndex++ );
            for( int v = 0; v < CellSeedType::getCornersCount(); v++ ) {
               iss >> vid;
               if( ! iss )
                  return false;
               seed.setCornerId( v, vid );
            }
         }
      }

      return meshBuilder.build( mesh );
   }

   String
   getMeshType() const
   {
      return "Meshes::Mesh";
   }

   int getMeshDimension() const
   {
      return this->meshDimension;
   }

   int
   getWorldDimension() const
   {
      return worldDimension;
   }

   VTKEntityType
   getCellVTKType() const
   {
      return cellVTKType;
   }

   String
   getRealType() const
   {
      return realType.c_str();
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
   // output of parseHeader
   std::string dataType;
   std::string dataset;

   String fileName;
   int meshDimension, worldDimension;
   VTKEntityType cellVTKType = VTKEntityType::Vertex;
   std::string realType;

   void reset()
   {
      fileName = "";
      meshDimension = worldDimension = 0;
      cellVTKType = VTKEntityType::Vertex;
      realType = "";
   }

   bool parseHeader( std::istream& str )
   {
      std::string line;
      std::istringstream iss;

      // check header
      getline( str, line );
      if( line != "# vtk DataFile Version 2.0" ) {
         std::cerr << "VTKReader: unsupported VTK header: '" << line << "'." << std::endl;
         return false;
      }

      // skip title
      if( ! str )
         return false;
      getline( str, line );

      // parse data type
      if( ! str )
         return false;
      getline( str, dataType );
      if( dataType != "ASCII" && dataType != "BINARY" ) {
         std::cerr << "VTKReader: unknown data type: '" << dataType << "'." << std::endl;
         return false;
      }

      // parse dataset
      if( ! str )
         return false;
      getline( str, line );
      iss.clear();
      iss.str( line );
      std::string tmp;
      iss >> tmp;
      if( tmp != "DATASET" ) {
         std::cerr << "VTKReader: wrong dataset specification: '" << line << "'." << std::endl;
         return false;
      }
      iss >> dataset;

      return true;
   }

   bool findSection( std::istream& str, const std::string& section, std::ios::pos_type begin = -1 )
   {
      std::string line, aux;
      std::istringstream iss;

      if( begin >= 0 )
         str.seekg( begin );

      while( str ) {
         std::ios::pos_type currentPosition = str.tellg();
         getline( str, line );
         iss.clear();
         iss.str( line );
         iss >> aux;
         if( aux == section ) {
            str.seekg( currentPosition );
            return true;
         }
      }

      return false;
   }
};

} // namespace Readers
} // namespace Meshes
} // namespace TNL
