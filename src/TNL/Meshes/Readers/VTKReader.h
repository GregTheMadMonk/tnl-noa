/***************************************************************************
                          VTKReader.h  -  description
                             -------------------
    begin                : Feb 25, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <fstream>
#include <sstream>
#include <vector>

#include <TNL/Meshes/Readers/MeshReader.h>
#include <TNL/Endianness.h>

namespace TNL {
namespace Meshes {
namespace Readers {

class VTKReader
: public MeshReader
{
public:
   VTKReader() = default;

   VTKReader( const std::string& fileName )
   : MeshReader( fileName )
   {}

   virtual void detectMesh() override
   {
      reset();

      std::ifstream inputFile( fileName );
      if( ! inputFile )
         throw MeshReaderError( "VTKReader", "failed to open the file '" + fileName + "'" );

      parseHeader( inputFile );

      if( dataset != "UNSTRUCTURED_GRID" )
         throw MeshReaderError( "VTKReader", "the dataset '" + dataset + "' is not supported" );

      // parse the file, find the starting positions of all relevant sections
      findSections( inputFile );

      std::string line, aux;
      std::istringstream iss;

      // parse points section
      if( ! sectionPositions.count( "POINTS" ) )
         throw MeshReaderError( "VTKReader", "unable to find the POINTS section, the file may be invalid or corrupted" );
      inputFile.seekg( sectionPositions["POINTS"] );
      getline( inputFile, line );
      iss.clear();
      iss.str( line );
      iss >> aux;
      iss >> NumberOfPoints;
      iss >> pointsType;
      if( pointsType != "float" && pointsType != "double" )
         throw MeshReaderError( "VTKReader", "unsupported data type for POINTS: " + pointsType );

      // global index type is not stored in legacy VTK files
      // (binary VTK files don't support int64)
      connectivityType = offsetsType = "std::int32_t";
      // only std::uint8_t makes sense for entity types
      typesType = "std::uint8_t";

      // arrays holding the data from the VTK file
      std::vector< double > pointsArray;
      std::vector< std::int32_t > connectivityArray, offsetsArray;
      std::vector< std::uint8_t > typesArray;

      // read points
      worldDimension = 0;
      for( std::size_t pointIndex = 0; pointIndex < NumberOfPoints; pointIndex++ ) {
         if( ! inputFile )
            throw MeshReaderError( "VTKReader", "unable to read enough vertices, the file may be invalid or corrupted" );

         // read the coordinates and compute the world dimension
         for( int i = 0; i < 3; i++ ) {
            double aux = 0;
            if( pointsType == "float" )
               aux = readValue< float >( dataFormat, inputFile );
            else
               aux = readValue< double >( dataFormat, inputFile );
            if( ! inputFile )
               throw MeshReaderError( "VTKReader", "unable to read " + std::to_string(i) + "th component of the vertex number " + std::to_string(pointIndex) );
            if( aux != 0.0 )
               worldDimension = std::max( worldDimension, i + 1 );
            pointsArray.push_back( aux );
         }
      }

      // skip to the CELL_TYPES section
      if( ! sectionPositions.count( "CELL_TYPES" ) )
         throw MeshReaderError( "VTKReader", "unable to find the CELL_TYPES section, the file may be invalid or corrupted" );
      inputFile.seekg( sectionPositions["CELL_TYPES"] );
      getline( inputFile, line );
      iss.clear();
      iss.str( line );
      iss >> aux;
      std::size_t NumberOfEntities = 0;
      iss >> NumberOfEntities;

      // read entity types
      for( std::size_t entityIndex = 0; entityIndex < NumberOfEntities; entityIndex++ ) {
         if( ! inputFile )
            throw MeshReaderError( "VTKReader", "unable to read enough cell types, the file may be invalid or corrupted" );
         // cell types are stored with great redundancy as int32 in the VTK file
         const std::uint8_t typeId = readValue< std::int32_t >( dataFormat, inputFile );
         typesArray.push_back( typeId );
      }

      // count entities for each dimension
      std::size_t entitiesCounts[4] = {0, 0, 0, 0};
      for( auto c : typesArray ) {
         const int dimension = getEntityDimension( (VTK::EntityShape) c );
         ++entitiesCounts[dimension];
      }

      // set meshDimension
      meshDimension = 3;
      if( entitiesCounts[3] == 0 ) {
         meshDimension--;
         if( entitiesCounts[2] == 0 ) {
            meshDimension--;
            if( entitiesCounts[1] == 0 )
               meshDimension--;
         }
      }

      if( meshDimension == 0 )
         throw MeshReaderError( "VTKReader", "Mesh dimension cannot be 0. Are there any entities at all?" );

      // filter out cell shapes
      std::vector< std::uint8_t > cellTypes;
      for( auto c : typesArray ) {
         const int dimension = getEntityDimension( (VTK::EntityShape) c );
         if( dimension == meshDimension )
            cellTypes.push_back( c );
      }

      // set number of cells
      NumberOfCells = cellTypes.size();
      if( NumberOfCells == 0 || NumberOfCells != entitiesCounts[meshDimension] ) {
         const std::string msg = "invalid number of cells (" + std::to_string(NumberOfCells) + "). Counts of entities for each dimension (0,1,2,3) are: "
                               + std::to_string(entitiesCounts[0]) + ", " + std::to_string(entitiesCounts[1]) + ", "
                               + std::to_string(entitiesCounts[2]) + ", " + std::to_string(entitiesCounts[3]);
         throw MeshReaderError( "VTKReader", msg );
      }

      // validate cell types
      cellShape = (VTK::EntityShape) cellTypes[0];
      for( auto c : cellTypes )
         if( (VTK::EntityShape) c != cellShape ) {
            const std::string msg = "Mixed unstructured meshes are not supported. There are cells with type "
                                  + VTK::getShapeName(cellShape) + " and " + VTK::getShapeName((VTK::EntityShape) c);
            throw MeshReaderError( "VTKReader", msg );
         }

      // find to the CELLS section
      if( ! sectionPositions.count( "CELLS" ) )
         throw MeshReaderError( "VTKReader", "unable to find the CELLS section, the file may be invalid or corrupted" );
      inputFile.seekg( sectionPositions["CELLS"] );
      getline( inputFile, line );

      // read entities
      for( std::size_t entityIndex = 0; entityIndex < NumberOfEntities; entityIndex++ ) {
         if( ! inputFile )
            throw MeshReaderError( "VTKReader", "unable to read enough cells, the file may be invalid or corrupted"
                                                " (entityIndex = " + std::to_string(entityIndex) + ")" );

         if( (VTK::EntityShape) typesArray[ entityIndex ] == cellShape ) {
            // read number of subvertices
            const std::int32_t subvertices = readValue< std::int32_t >( dataFormat, inputFile );
            for( int v = 0; v < subvertices; v++ ) {
               // legacy VTK files do not support 64-bit integers, even in the BINARY format
               const std::int32_t vid = readValue< std::int32_t >( dataFormat, inputFile );
               if( ! inputFile )
                  throw MeshReaderError( "VTKReader", "unable to read enough cells, the file may be invalid or corrupted"
                                                      " (entityIndex = " + std::to_string(entityIndex) + ", subvertex = " + std::to_string(v) + ")" );
               connectivityArray.push_back( vid );
            }
            offsetsArray.push_back( connectivityArray.size() );
         }
      }

      // set cell types
      std::swap( cellTypes, typesArray );

      // set the arrays to the base class
      this->pointsArray = std::move(pointsArray);
      this->connectivityArray = std::move(connectivityArray);
      this->offsetsArray = std::move(offsetsArray);
      this->typesArray = std::move(typesArray);

      // indicate success by setting the mesh type
      meshType = "Meshes::Mesh";
   }

   virtual void reset() override
   {
      resetBase();
      dataFormat = VTK::FileFormat::ascii;
      dataset = "";
      sectionPositions.clear();
   }

protected:
   // output of parseHeader
   VTK::FileFormat dataFormat = VTK::FileFormat::ascii;
   std::string dataset;

   // output of findSections
   std::map< std::string, std::ios::pos_type > sectionPositions;

   void parseHeader( std::istream& str )
   {
      std::string line;
      std::istringstream iss;

      // check header
      getline( str, line );
      if( line != "# vtk DataFile Version 2.0" )
         throw MeshReaderError( "VTKReader", "failed to parse the VTK file header: unsupported VTK header '" + line + "'" );

      // skip title
      if( ! str )
         throw MeshReaderError( "VTKReader", "failed to parse the VTK file header" );
      getline( str, line );

      // parse data type
      if( ! str )
         throw MeshReaderError( "VTKReader", "failed to parse the VTK file header" );
      std::string format;
      getline( str, format );
      if( format == "ASCII" )
         dataFormat = VTK::FileFormat::ascii;
      else if( format == "BINARY" )
         dataFormat = VTK::FileFormat::binary;
      else
         throw MeshReaderError( "VTKReader", "unknown data format: '" + format + "'" );

      // parse dataset
      if( ! str )
         throw MeshReaderError( "VTKReader", "failed to parse the VTK file header" );
      getline( str, line );
      iss.clear();
      iss.str( line );
      std::string tmp;
      iss >> tmp;
      if( tmp != "DATASET" )
         throw MeshReaderError( "VTKReader", "wrong dataset specification: '" + line + "'" );
      iss >> dataset;
   }

   void findSections( std::istream& str )
   {
      while( str ) {
         // drop all whitespace (empty lines etc) before saving a position and reading a line
         str >> std::ws;
         if( str.eof() )
            break;

         // read a line which should contain the following section header
         const std::ios::pos_type currentPosition = str.tellg();
         std::string line;
         getline( str, line );
         if( ! str )
            throw MeshReaderError( "VTKReader", "failed to parse sections of the VTK file" );

         // parse the section name
         std::istringstream iss( line );
         std::string name;
         iss >> name;

         if( name == "FIELD" ) {
            sectionPositions.insert( {"FIELD", currentPosition} );
            // parse the rest of the line: FIELD FieldData <count>
            std::string aux;
            int count = 0;
            iss >> aux >> count;
            // skip the FieldData arrays
            for( int i = 0; i < count; i++ ) {
               getline( str, line );
               iss.clear();
               iss.str( line );
               // <name> <components> <tuples> <datatype>
               std::int32_t components = 0;
               std::int32_t tuples = 0;
               std::string datatype;
               iss >> aux >> components >> tuples >> datatype;
               if( ! iss )
                  throw MeshReaderError( "VTKReader", "failed to extract FieldData information from line '" + line + "'" );
               // skip the points coordinates
               for( std::int32_t j = 0; j < components * tuples; j++ )
                  skipValue( dataFormat, str, datatype );
            }
         }
         else if( name == "POINTS" ) {
            sectionPositions.insert( {"POINTS", currentPosition} );
            // parse the rest of the line: POINTS <count> <datatype>
            std::int32_t count = 0;
            std::string datatype;
            iss >> count >> datatype;
            // skip the values
            for( std::int32_t j = 0; j < 3 * count; j++ )
               skipValue( dataFormat, str, datatype );
         }
         else if( name == "CELLS" ) {
            sectionPositions.insert( {"CELLS", currentPosition} );
            // parse the rest of the line: CELLS <cells_count> <values_count>
            std::int32_t cells_count = 0;
            std::int32_t values_count = 0;
            iss >> cells_count >> values_count;
            // skip the values
            for( std::int32_t j = 0; j < values_count; j++ )
               skipValue( dataFormat, str, "int" );
         }
         else if( name == "CELL_TYPES" ) {
            sectionPositions.insert( {"CELL_TYPES", currentPosition} );
            // parse the rest of the line: CELL_TYPES <count>
            std::int32_t count = 0;
            iss >> count;
            // skip the values
            for( std::int32_t j = 0; j < count; j++ )
               // cell types are stored with great redundancy as int32 in the VTK file
               skipValue( dataFormat, str, "int" );
         }
         else if( name == "CELL_DATA" || name == "POINT_DATA" ) {
            // TODO: implement reading values from these sections
            break;
         }
         else
            throw MeshReaderError( "VTKReader", "parsing error: unexpected section start at byte " + std::to_string(currentPosition)
                                    + " (section name is '" + name + "')" );
      }
   }

   static void skipValue( VTK::FileFormat format, std::istream& str, std::string datatype )
   {
      if( datatype == "int" )
         readValue< std::int32_t >( format, str );
      else if( datatype == "float" )
         readValue< float >( format, str );
      else if( datatype == "double" )
         readValue< double >( format, str );
      else
         throw MeshReaderError( "VTKReader", "found data type which is not implemented in the reader: " + datatype );
   }

   template< typename T >
   static T readValue( VTK::FileFormat format, std::istream& str )
   {
      T value;
      if( format == VTK::FileFormat::binary ) {
         str.read( reinterpret_cast<char*>(&value), sizeof(T) );
         // forceBigEndian = swapIfLittleEndian, i.e. here it forces a big-endian
         // value to the correct system endianness
         value = forceBigEndian( value );
      }
      else {
         str >> value;
      }
      return value;
   }
};

} // namespace Readers
} // namespace Meshes
} // namespace TNL
