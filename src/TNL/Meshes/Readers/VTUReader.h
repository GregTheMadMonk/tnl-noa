/***************************************************************************
                          VTUReader.h  -  description
                             -------------------
    begin                : Mar 21, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <map>
#include <set>
#include <TNL/variant.hpp>

#include <TNL/Meshes/MeshBuilder.h>
#include <TNL/Meshes/VTKTraits.h>
#include <TNL/base64.h>
#include <TNL/Endianness.h>

#ifdef HAVE_ZLIB
   #include <TNL/zlib_compression.h>
#endif

#ifdef HAVE_TINYXML2
   #include <tinyxml2.h>
#endif

namespace TNL {
namespace Meshes {
namespace Readers {

struct XMLVTKError
   : public std::runtime_error
{
   XMLVTKError( std::string msg )
   : std::runtime_error( "XMLVTKReader error: " + msg )
   {}
};

static const std::map< std::string, std::string > VTKDataTypes {
   {"Int8", "std::int8_t"},
   {"UInt8", "std::uint8_t"},
   {"Int16", "std::int16_t"},
   {"UInt16", "std::uint16_t"},
   {"Int32", "std::int32_t"},
   {"UInt32", "std::uint32_t"},
   {"Int64", "std::int64_t"},
   {"UInt64", "std::uint64_t"},
   {"Float32", "float"},
   {"Float64", "double"}
};

class VTUReader
{
   using VariantVector = mpark::variant< std::vector< std::int8_t >,
                                         std::vector< std::uint8_t >,
                                         std::vector< std::int16_t >,
                                         std::vector< std::uint16_t >,
                                         std::vector< std::int32_t >,
                                         std::vector< std::uint32_t >,
                                         std::vector< std::int64_t >,
                                         std::vector< std::uint64_t >,
                                         std::vector< float >,
                                         std::vector< double > >;

#ifdef HAVE_TINYXML2
   static void verifyElement( const tinyxml2::XMLElement* elem, const std::string name )
   {
      if( ! elem )
         throw XMLVTKError( "tag <" + name + "> not found" );
      if( elem->Name() != name )
         throw XMLVTKError( "invalid XML format - expected a <" + name + "> element, got <" + elem->Name() + ">" );
   }

   static const tinyxml2::XMLElement*
   verifyHasOnlyOneChild( const tinyxml2::XMLElement* parent, const std::string childName = "" )
   {
      const std::string parentName = parent->Name();
      const tinyxml2::XMLElement* elem = parent->FirstChildElement();
      if( ! childName.empty() )
         verifyElement( elem, childName );
      else if( ! elem )
         throw XMLVTKError( "element " + parentName + " does not contain any child" );
      if( elem->NextSibling() )
         throw XMLVTKError( "<" + childName + "> is not the only element in <" + parentName + ">" );
      return elem;
   }

   static std::string
   getAttributeString( const tinyxml2::XMLElement* elem, std::string name, std::string defaultValue = "" )
   {
      const char* attribute = nullptr;
      attribute = elem->Attribute( name.c_str() );
      if( attribute )
         return attribute;
      if( ! defaultValue.empty() )
         return defaultValue;
      throw XMLVTKError( "element <" + std::string(elem->Name()) + "> does not have the attribute '" + name + "'" );
   }

   static std::int64_t
   getAttributeInteger( const tinyxml2::XMLElement* elem, std::string name )
   {
      std::int64_t value;
      tinyxml2::XMLError status = elem->QueryInt64Attribute( name.c_str(), &value );
      if( status != tinyxml2::XML_SUCCESS )
         throw XMLVTKError( "element <" + std::string(elem->Name()) + "> does not have the attribute '" + name + "' or it could not be converted to int64_t" );
      return value;
   }

   static const tinyxml2::XMLElement*
   getChildSafe( const tinyxml2::XMLElement* parent, std::string name )
   {
      const tinyxml2::XMLElement* child = parent->FirstChildElement( name.c_str() );
      verifyElement( child, name );
      return child;
   }

   static void
   verifyDataArray( const tinyxml2::XMLElement* elem )
   {
      verifyElement( elem, "DataArray" );
      // verify Name
      getAttributeString( elem, "Name" );
      // verify type
      const std::string type = getAttributeString( elem, "type" );
      if( VTKDataTypes.count( type ) == 0 )
         throw XMLVTKError( "unsupported DataArray type: " + type );
      // verify format
      const std::string format = getAttributeString( elem, "format" );
      if( format != "ascii" && format != "binary" )
         throw XMLVTKError( "unsupported DataArray format: " + format );
      // verify NumberOfComponents (optional)
      const std::string NumberOfComponents = getAttributeString( elem, "NumberOfComponents", "0" );
      static const std::set< std::string > validNumbersOfComponents = {"0", "1", "2", "3"};
      if( validNumbersOfComponents.count( NumberOfComponents ) == 0 )
         throw XMLVTKError( "unsupported NumberOfComponents in DataArray: " + NumberOfComponents );
   }

   static const tinyxml2::XMLElement*
   getDataArrayByName( const tinyxml2::XMLElement* parent, std::string name )
   {
      const tinyxml2::XMLElement* found = nullptr;
      const tinyxml2::XMLElement* child = parent->FirstChildElement( "DataArray" );
      while( child != nullptr ) {
         verifyElement( child, "DataArray" );
         std::string arrayName;
         try {
            arrayName = getAttributeString( child, "Name" );
         }
         catch( const XMLVTKError& ) {}
         if( arrayName == name ) {
            if( found == nullptr )
               found = child;
            else
               throw XMLVTKError( "the <" + std::string(parent->Name()) + "> tag contains multiple <DataArray> tags with the Name=\"" + name + "\" attribute" );
         }
         child = child->NextSiblingElement( "DataArray" );
      }
      if( found == nullptr )
         throw XMLVTKError( "the <" + std::string(parent->Name()) + "> tag does not contain any <DataArray> tag with the Name=\"" + name + "\" attribute" );
      verifyDataArray( found );
      return found;
   }

   template< typename HeaderType >
   std::size_t
   readBlockSize( const char* block ) const
   {
      std::pair<std::size_t, std::unique_ptr<char[]>> decoded_data = decode_block( block, get_encoded_length(sizeof(HeaderType)) );
      if( decoded_data.first != sizeof(HeaderType) )
         throw XMLVTKError( "base64-decoding failed - mismatched data size in the binary header (read "
                            + std::to_string(decoded_data.first) + " bytes, expected " + std::to_string(sizeof(HeaderType)) + " bytes)" );
      const HeaderType* blockSize = reinterpret_cast<const HeaderType*>(decoded_data.second.get());
      return *blockSize;
   }

   template< typename HeaderType, typename T >
   VariantVector
   readBinaryBlock( const char* block ) const
   {
      // skip whitespace at the beginning
      while( *block != '\0' && std::isspace( *block ) )
         ++block;

      if( compressor == "" ) {
         const std::size_t blockSize = readBlockSize< HeaderType >( block );
         block += get_encoded_length(sizeof(HeaderType));
         std::pair<std::size_t, std::unique_ptr<char[]>> decoded_data = decode_block( block, get_encoded_length(blockSize) );
         std::vector<T> vector( decoded_data.first / sizeof(T) );
         for( std::size_t i = 0; i < vector.size(); i++ )
            vector[i] = reinterpret_cast<const T*>(decoded_data.second.get())[i];
         return vector;
      }
      else if( compressor == "vtkZLibDataCompressor" ) {
#ifdef HAVE_ZLIB
         std::pair<HeaderType, std::unique_ptr<T[]>> decoded_data = decompress_block< HeaderType, T >(block);
         std::vector<T> vector( decoded_data.first );
         for( std::size_t i = 0; i < vector.size(); i++ )
            vector[i] = decoded_data.second.get()[i];
         return vector;
#else
         throw XMLVTKError( "The ZLIB compression is not available in this build. Make sure that ZLIB is "
                            "installed and recompile the program with -DHAVE_ZLIB." );
#endif
      }
      else
         throw XMLVTKError( "unsupported compressor type: " + compressor + " (only vtkZLibDataCompressor is supported)" );
   }

   template< typename T >
   VariantVector
   readBinaryBlock( const char* block ) const
   {
      if( headerType == "std::int8_t" )          return readBinaryBlock< std::int8_t,   T >( block );
      else if( headerType == "std::uint8_t" )    return readBinaryBlock< std::uint8_t,  T >( block );
      else if( headerType == "std::int16_t" )    return readBinaryBlock< std::int16_t,  T >( block );
      else if( headerType == "std::uint16_t" )   return readBinaryBlock< std::uint16_t, T >( block );
      else if( headerType == "std::int32_t" )    return readBinaryBlock< std::int32_t,  T >( block );
      else if( headerType == "std::uint32_t" )   return readBinaryBlock< std::uint32_t, T >( block );
      else if( headerType == "std::int64_t" )    return readBinaryBlock< std::int64_t,  T >( block );
      else if( headerType == "std::uint64_t" )   return readBinaryBlock< std::uint64_t, T >( block );
      else throw XMLVTKError( "unsupported header type: " + headerType );
   }

   VariantVector
   readDataArray( const tinyxml2::XMLElement* elem, std::string arrayName ) const
   {
      verifyElement( elem, "DataArray" );
      const char* block = elem->GetText();
      if( ! block )
         throw XMLVTKError( "the DataArray with Name=\"" + arrayName + "\" does not contain any data" );
      const std::string type = getAttributeString( elem, "type" );
      const std::string format = getAttributeString( elem, "format" );
      if( format == "ascii" ) {
         // TODO
         throw XMLVTKError( "reading ASCII arrays is not implemented yet" );
      }
      else if( format == "binary" ) {
         if( type == "Int8" )          return readBinaryBlock< std::int8_t   >( block );
         else if( type == "UInt8" )    return readBinaryBlock< std::uint8_t  >( block );
         else if( type == "Int16" )    return readBinaryBlock< std::int16_t  >( block );
         else if( type == "UInt16" )   return readBinaryBlock< std::uint16_t >( block );
         else if( type == "Int32" )    return readBinaryBlock< std::int32_t  >( block );
         else if( type == "UInt32" )   return readBinaryBlock< std::uint32_t >( block );
         else if( type == "Int64" )    return readBinaryBlock< std::int64_t  >( block );
         else if( type == "UInt64" )   return readBinaryBlock< std::uint64_t >( block );
         else if( type == "Float32" )  return readBinaryBlock< float  >( block );
         else if( type == "Float64" )  return readBinaryBlock< double >( block );
         else throw XMLVTKError( "unsupported DataArray type: " + type );
      }
      else
         throw XMLVTKError( "unsupported DataArray format: " + format );
   }

   void readUnstructuredGrid( const tinyxml2::XMLElement* elem )
   {
      using namespace tinyxml2;
      const XMLElement* piece = getChildSafe( elem, "Piece" );
      if( piece->NextSiblingElement( "Piece" ) )
         // ambiguity - throw error, we don't know which piece to parse (or all of them?)
         throw XMLVTKError( "the serial UnstructuredGrid file contains more than one <Piece> element" );
      NumberOfPoints = getAttributeInteger( piece, "NumberOfPoints" );
      NumberOfCells = getAttributeInteger( piece, "NumberOfCells" );

      // verify points
      const XMLElement* points = getChildSafe( piece, "Points" );
      const XMLElement* pointsData = verifyHasOnlyOneChild( points, "DataArray" );
      verifyDataArray( pointsData );
      const std::string pointsDataName = getAttributeString( pointsData, "Name" );
      if( pointsDataName != "Points" )
         throw XMLVTKError( "the <Points> tag does not contain a <DataArray> with Name=\"Points\" attribute" );

      // verify cells
      const XMLElement* cells = getChildSafe( piece, "Cells" );
      const XMLElement* connectivity = getDataArrayByName( cells, "connectivity" );
      const XMLElement* offsets = getDataArrayByName( cells, "offsets" );
      const XMLElement* types = getDataArrayByName( cells, "types" );

      // read the points, connectivity, offsets and types into intermediate arrays
      pointsArray = readDataArray( pointsData, "Points" );
      pointsType = VTKDataTypes.at( getAttributeString( pointsData, "type" ) );
      connectivityArray = readDataArray( connectivity, "connectivity" );
      connectivityType = VTKDataTypes.at( getAttributeString( connectivity, "type" ) );
      offsetsArray = readDataArray( offsets, "offsets" );
      offsetsType = VTKDataTypes.at( getAttributeString( offsets, "type" ) );
      typesArray = readDataArray( types, "types" );
      typesType = VTKDataTypes.at( getAttributeString( types, "type" ) );

      // connectivity and offsets must have the same type
      if( connectivityType != offsetsType )
         throw XMLVTKError( "the \"connectivity\" and \"offsets\" array do not have the same type ("
                            + connectivityType + " vs " + offsetsType + ")" );
      // cell types can be only uint8_t
      if( typesType != "std::uint8_t" )
         throw XMLVTKError( "unsupported data type for the Name=\"types\" array" );

      using mpark::visit;
      // validate points
      visit( [this](auto&& array) {
               // check array size
               if( array.size() != 3 * NumberOfPoints )
                  throw XMLVTKError( "invalid size of the Points data array (" + std::to_string(array.size())
                                     + " vs " + std::to_string(NumberOfPoints) + ")" );
               // set worldDimension
               worldDimension = 1;
               std::size_t i = 0;
               for( auto c : array ) {
                  if( c != 0 ) {
                     int dim = i % 3 + 1;
                     worldDimension = std::max( worldDimension, dim );
                  }
                  ++i;
               }
            },
            pointsArray
         );
      // validate types
      visit( [this](auto&& array) {
               // check array size
               if( array.size() != NumberOfCells )
                  throw XMLVTKError( "size of the types data array does not match the NumberOfCells attribute" );
               cellShape = (VTK::EntityShape) array[0];
               meshDimension = getEntityDimension( cellShape );
               // TODO: check only entities of the same dimension (edges, faces and cells separately)
               for( auto c : array )
                  if( (VTK::EntityShape) c != cellShape )
                     throw XMLVTKError( "Mixed unstructured meshes are not supported. There are cells with type "
                                        + VTK::getShapeName(cellShape) + " and " + VTK::getShapeName((VTK::EntityShape) c) + "." );
            },
            typesArray
         );
      // validate offsets
      std::size_t max_offset = 0;
      visit( [this, &max_offset](auto&& array) mutable {
               if( array.size() != NumberOfCells )
                  throw XMLVTKError( "size of the offsets data array does not match the NumberOfCells attribute" );
               for( auto c : array ) {
                  if( c <= (decltype(c)) max_offset )
                     throw XMLVTKError( "the offsets array is not monotonically increasing" );
                  max_offset = c;
               }
            },
            offsetsArray
         );
      // validate connectivity
      visit( [this, max_offset](auto&& array) {
               if( array.size() != max_offset )
                  throw XMLVTKError( "size of the connectivity data array does not match the offsets array" );
               for( auto c : array ) {
                  if( c < 0 || (std::size_t) c >= NumberOfPoints )
                     throw XMLVTKError( "connectivity index " + std::to_string(c) + " is out of range" );
               }
            },
            connectivityArray
         );

   }
#endif

public:
   VTUReader() = delete;

   VTUReader( const std::string& fileName )
   : fileName( fileName )
   {}

   bool detectMesh()
   {
#ifdef HAVE_TINYXML2
      this->reset();

      using namespace tinyxml2;

      XMLDocument dom;
      XMLError status;

      // load and verify XML
      status = dom.LoadFile( fileName.c_str() );
      if( status != XML_SUCCESS )
         throw XMLVTKError( "VTUReader: failed to parse the file as an XML document." );

      // verify root element
      const XMLElement* elem = dom.FirstChildElement();
      verifyElement( elem, "VTKFile" );
      if( elem->NextSibling() )
         throw XMLVTKError( "<VTKFile> is not the only element in the file" );

      // verify byte order
      const std::string systemByteOrder = (isLittleEndian()) ? "LittleEndian" : "BigEndian";
      byteOrder = getAttributeString( elem, "byte_order" );
      if( byteOrder != systemByteOrder )
         throw XMLVTKError( "incompatible byte_order: " + byteOrder + " (the system is " + systemByteOrder + " and the conversion "
                            "from BigEndian to LittleEndian or vice versa is not implemented yet)" );

      // verify header type
      headerType = getAttributeString( elem, "header_type", "UInt32" );
      if( VTKDataTypes.count( headerType ) == 0 )
         throw XMLVTKError( "invalid header_type: " + headerType );
      headerType = VTKDataTypes.at( headerType );

      // verify compressor
      compressor = getAttributeString( elem, "compressor", "<none>" );
      if( compressor == "<none>" )
         compressor = "";
      if( compressor != "" && compressor != "vtkZLibDataCompressor" )
         throw XMLVTKError( "unsupported compressor type: " + compressor + " (only vtkZLibDataCompressor is supported)" );

      // verify file type
      fileType = getAttributeString( elem, "type" );
      elem = verifyHasOnlyOneChild( elem, fileType );
      if( fileType == "UnstructuredGrid" )
         readUnstructuredGrid( elem );
      else
         // TODO: generalize the reader for other XML VTK formats
         throw XMLVTKError( "parsing the " + fileType + " files is not implemented (yet)" );

      return true;
#else
      throw std::runtime_error("The program was compiled without XML parsing. Make sure that TinyXML-2 is "
                               "installed and recompile the program with -DHAVE_TINYXML2.");
#endif
   }

   template< typename MeshType >
   bool readMesh( MeshType& mesh )
   {
      // TODO: check that detectMesh has been called

      // check that the cell shape mathes
      const VTK::EntityShape meshCellShape = VTK::TopologyToEntityShape< typename MeshType::template EntityTraits< MeshType::getMeshDimension() >::EntityTopology >::shape;
      if( meshCellShape != cellShape )
         throw XMLVTKError( "the mesh cell shape " + VTK::getShapeName(meshCellShape) + " does not match the shape "
                            + "of cells used in the file (" + VTK::getShapeName(cellShape) + ")" );

      using MeshBuilder = MeshBuilder< MeshType >;
      using IndexType = typename MeshType::GlobalIndexType;
      using PointType = typename MeshType::PointType;
      using CellSeedType = typename MeshBuilder::CellSeedType;

      MeshBuilder meshBuilder;
      meshBuilder.setPointsCount( NumberOfPoints );
      meshBuilder.setCellsCount( NumberOfCells );

      // assign points
      visit( [this, &meshBuilder](auto&& array) {
               PointType p;
               std::size_t i = 0;
               for( auto c : array ) {
                  int dim = i++ % 3;
                  if( dim >= PointType::getSize() )
                     continue;
                  p[dim] = c;
                  if( dim == PointType::getSize() - 1 )
                     meshBuilder.setPoint( (i - 1) / 3, p );
               }
            },
            pointsArray
         );

      // assign cells
      visit( [this, &meshBuilder](auto&& connectivity) {
               // let's just assume that the connectivity and offsets arrays have the same type...
               using mpark::get;
               const auto& offsets = get< std::decay_t<decltype(connectivity)> >( offsetsArray );
               std::size_t offsetStart = 0;
               for( std::size_t i = 0; i < NumberOfCells; i++ ) {
                  CellSeedType& seed = meshBuilder.getCellSeed( i );
                  const std::size_t offsetEnd = offsets[ i ];
                  for( std::size_t o = offsetStart; o < offsetEnd; o++ )
                     seed.setCornerId( o - offsetStart, connectivity[ o ] );
                  offsetStart = offsetEnd;
               }
            },
            connectivityArray
         );

      // reset arrays since they are not needed anymore
      pointsArray = connectivityArray = offsetsArray = typesArray = {};

      return meshBuilder.build( mesh );
   }

   std::string
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

   VTK::EntityShape
   getCellShape() const
   {
      return cellShape;
   }

   std::string
   getRealType() const
   {
      return pointsType.c_str();
   }

   std::string
   getGlobalIndexType() const
   {
      return connectivityType;
   }

   std::string
   getLocalIndexType() const
   {
      // not stored in the VTK file
      return "short int";
   }

protected:
   std::string fileName;

   // VTK file type
   std::string fileType;

   std::size_t NumberOfPoints, NumberOfCells;
   int meshDimension, worldDimension;
   VTK::EntityShape cellShape = VTK::EntityShape::Vertex;
   std::string byteOrder, compressor, headerType;

   // arrays holding the <DataArray>s from the VTK file
   VariantVector pointsArray, connectivityArray, offsetsArray, typesArray;
   // string representation of each array's value type
   std::string pointsType, connectivityType, offsetsType, typesType;

   void reset()
   {
      fileType = "";
      NumberOfPoints = NumberOfCells = 0;
      meshDimension = worldDimension = 0;
      cellShape = VTK::EntityShape::Vertex;
      byteOrder = compressor = headerType = "";
      pointsArray = connectivityArray = offsetsArray = typesArray = {};
      pointsType = connectivityType = offsetsType = typesType = "";
   }
};

} // namespace Readers
} // namespace Meshes
} // namespace TNL
