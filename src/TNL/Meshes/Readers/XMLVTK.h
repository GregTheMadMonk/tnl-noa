/***************************************************************************
                          XMLVTK.h  -  description
                             -------------------
    begin                : Mar 21, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <map>
#include <set>
#include <experimental/filesystem>

#include <TNL/Meshes/Readers/MeshReader.h>
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

class XMLVTK
: public MeshReader
{
#ifdef HAVE_TINYXML2
protected:
   static void verifyElement( const tinyxml2::XMLElement* elem, const std::string name )
   {
      if( ! elem )
         throw MeshReaderError( "XMLVTK", "tag <" + name + "> not found" );
      if( elem->Name() != name )
         throw MeshReaderError( "XMLVTK", "invalid XML format - expected a <" + name + "> element, got <" + elem->Name() + ">" );
   }

   static const tinyxml2::XMLElement*
   verifyHasOnlyOneChild( const tinyxml2::XMLElement* parent, const std::string childName = "" )
   {
      const std::string parentName = parent->Name();
      const tinyxml2::XMLElement* elem = parent->FirstChildElement();
      if( ! childName.empty() )
         verifyElement( elem, childName );
      else if( ! elem )
         throw MeshReaderError( "XMLVTK", "element " + parentName + " does not contain any child" );
      if( elem->NextSibling() )
         throw MeshReaderError( "XMLVTK", "<" + childName + "> is not the only element in <" + parentName + ">" );
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
      throw MeshReaderError( "XMLVTK", "element <" + std::string(elem->Name()) + "> does not have the attribute '" + name + "'" );
   }

   static std::int64_t
   getAttributeInteger( const tinyxml2::XMLElement* elem, std::string name )
   {
      std::int64_t value;
      tinyxml2::XMLError status = elem->QueryInt64Attribute( name.c_str(), &value );
      if( status != tinyxml2::XML_SUCCESS )
         throw MeshReaderError( "XMLVTK", "element <" + std::string(elem->Name()) + "> does not have the attribute '" + name + "' or it could not be converted to int64_t" );
      return value;
   }

   static std::int64_t
   getAttributeInteger( const tinyxml2::XMLElement* elem, std::string name, int64_t defaultValue )
   {
      std::int64_t value;
      tinyxml2::XMLError status = elem->QueryInt64Attribute( name.c_str(), &value );
      if( status != tinyxml2::XML_SUCCESS )
         return defaultValue;
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
   verifyDataArray( const tinyxml2::XMLElement* elem, std::string elemName = "DataArray" )
   {
      // the elemName parameter is necessary due to parallel formats using "PDataArray"
      verifyElement( elem, elemName.c_str() );
      // verify Name
      getAttributeString( elem, "Name" );
      // verify type
      const std::string type = getAttributeString( elem, "type" );
      if( VTKDataTypes.count( type ) == 0 )
         throw MeshReaderError( "XMLVTK", "unsupported " + elemName + " type: " + type );
      // verify format (attribute not used on PDataArray)
      if( elemName == "DataArray" ) {
         const std::string format = getAttributeString( elem, "format" );
         if( format != "ascii" && format != "binary" )
            throw MeshReaderError( "XMLVTK", "unsupported " + elemName + " format: " + format );
      }
      // verify NumberOfComponents (optional)
      const std::string NumberOfComponents = getAttributeString( elem, "NumberOfComponents", "0" );
      static const std::set< std::string > validNumbersOfComponents = {"0", "1", "2", "3"};
      if( validNumbersOfComponents.count( NumberOfComponents ) == 0 )
         throw MeshReaderError( "XMLVTK", "unsupported NumberOfComponents in " + elemName + ": " + NumberOfComponents );
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
         catch( const MeshReaderError& ) {}
         if( arrayName == name ) {
            if( found == nullptr )
               found = child;
            else
               throw MeshReaderError( "XMLVTK", "the <" + std::string(parent->Name()) + "> tag contains multiple <DataArray> tags with the Name=\"" + name + "\" attribute" );
         }
         child = child->NextSiblingElement( "DataArray" );
      }
      if( found == nullptr )
         throw MeshReaderError( "XMLVTK", "the <" + std::string(parent->Name()) + "> tag does not contain any <DataArray> tag with the Name=\"" + name + "\" attribute" );
      verifyDataArray( found );
      return found;
   }

   template< typename T >
   VariantVector
   readAsciiBlock( const char* block ) const
   {
      // creating a copy of the block is rather costly, but so is ASCII parsing
      std::stringstream ss;
      ss << block;

      std::vector<T> vector;
      while( ss ) {
         // since std::uint8_t is an alias to unsigned char, we need to parse
         // bytes into a larger type, otherwise operator>> would read it as char
         std::common_type_t< T, std::uint16_t > value;
         ss >> value;
         if( ss )
            vector.push_back( value );
      }

      return vector;
   }

   template< typename HeaderType >
   static std::size_t
   readBlockSize( const char* block )
   {
      std::pair<std::size_t, std::unique_ptr<std::uint8_t[]>> decoded_data = base64::decode( block, base64::get_encoded_length(sizeof(HeaderType)) );
      if( decoded_data.first != sizeof(HeaderType) )
         throw MeshReaderError( "XMLVTK", "base64-decoding failed - mismatched data size in the binary header (read "
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
         block += base64::get_encoded_length(sizeof(HeaderType));
         std::pair<std::size_t, std::unique_ptr<std::uint8_t[]>> decoded_data = base64::decode( block, base64::get_encoded_length(blockSize) );
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
         throw MeshReaderError( "XMLVTK", "The ZLIB compression is not available in this build. Make sure that ZLIB is "
                                          "installed and recompile the program with -DHAVE_ZLIB." );
#endif
      }
      else
         throw MeshReaderError( "XMLVTK", "unsupported compressor type: " + compressor + " (only vtkZLibDataCompressor is supported)" );
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
      else throw MeshReaderError( "XMLVTK", "unsupported header type: " + headerType );
   }

   VariantVector
   readDataArray( const tinyxml2::XMLElement* elem, std::string arrayName ) const
   {
      verifyElement( elem, "DataArray" );
      const char* block = elem->GetText();
      if( ! block )
         throw MeshReaderError( "XMLVTK", "the DataArray with Name=\"" + arrayName + "\" does not contain any data" );
      const std::string type = getAttributeString( elem, "type" );
      const std::string format = getAttributeString( elem, "format" );
      if( format == "ascii" ) {
         if( type == "Int8" )          return readAsciiBlock< std::int8_t   >( block );
         else if( type == "UInt8" )    return readAsciiBlock< std::uint8_t  >( block );
         else if( type == "Int16" )    return readAsciiBlock< std::int16_t  >( block );
         else if( type == "UInt16" )   return readAsciiBlock< std::uint16_t >( block );
         else if( type == "Int32" )    return readAsciiBlock< std::int32_t  >( block );
         else if( type == "UInt32" )   return readAsciiBlock< std::uint32_t >( block );
         else if( type == "Int64" )    return readAsciiBlock< std::int64_t  >( block );
         else if( type == "UInt64" )   return readAsciiBlock< std::uint64_t >( block );
         else if( type == "Float32" )  return readAsciiBlock< float  >( block );
         else if( type == "Float64" )  return readAsciiBlock< double >( block );
         else throw MeshReaderError( "XMLVTK", "unsupported DataArray type: " + type );
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
         else throw MeshReaderError( "XMLVTK", "unsupported DataArray type: " + type );
      }
      else
         throw MeshReaderError( "XMLVTK", "unsupported DataArray format: " + format );
   }

   VariantVector
   readPointOrCellData( std::string sectionName, std::string arrayName )
   {
      const tinyxml2::XMLElement* piece = getChildSafe( datasetElement, "Piece" );
      if( piece->NextSiblingElement( "Piece" ) )
         // ambiguity - throw error, we don't know which piece to parse
         throw MeshReaderError( "XMLVTK", "the dataset element <" + fileType + "> contains more than one <Piece> element" );
      const tinyxml2::XMLElement* pointData = getChildSafe( piece, sectionName.c_str() );
      if( pointData->NextSiblingElement( sectionName.c_str() ) )
         throw MeshReaderError( "XMLVTK", "the <Piece> element contains more than one <" + sectionName + "> element" );
      const tinyxml2::XMLElement* dataArray = getDataArrayByName( pointData, arrayName.c_str() );
      return readDataArray( dataArray, arrayName.c_str() );
   }
#endif

protected:
   [[noreturn]] static void
   throw_no_tinyxml()
   {
      throw std::runtime_error("The program was compiled without XML parsing. Make sure that TinyXML-2 is "
                               "installed and recompile the program with -DHAVE_TINYXML2.");
   }

public:
   XMLVTK() = default;

   XMLVTK( const std::string& fileName )
   : MeshReader( fileName )
   {}

   void openVTKFile()
   {
#ifdef HAVE_TINYXML2
      using namespace tinyxml2;

      namespace fs = std::experimental::filesystem;
      if( ! fs::exists( fileName ) )
         throw MeshReaderError( "XMLVTK", "file '" + fileName + "' does not exist" );
      if( fs::is_directory( fileName ) )
         throw MeshReaderError( "XMLVTK", "path '" + fileName + "' is a directory" );

      // load and verify XML
      tinyxml2::XMLError status = dom.LoadFile( fileName.c_str() );
      if( status != XML_SUCCESS )
         throw MeshReaderError( "XMLVTK", "failed to parse the file as an XML document." );

      // verify root element
      const XMLElement* elem = dom.FirstChildElement();
      verifyElement( elem, "VTKFile" );
      if( elem->NextSibling() )
         throw MeshReaderError( "XMLVTK", "<VTKFile> is not the only element in the file" );

      // verify byte order
      const std::string systemByteOrder = (isLittleEndian()) ? "LittleEndian" : "BigEndian";
      byteOrder = getAttributeString( elem, "byte_order" );
      if( byteOrder != systemByteOrder )
         throw MeshReaderError( "XMLVTK", "incompatible byte_order: " + byteOrder + " (the system is " + systemByteOrder + " and the conversion "
                                          "from BigEndian to LittleEndian or vice versa is not implemented yet)" );

      // verify header type
      headerType = getAttributeString( elem, "header_type", "UInt32" );
      if( VTKDataTypes.count( headerType ) == 0 )
         throw MeshReaderError( "XMLVTK", "invalid header_type: " + headerType );
      headerType = VTKDataTypes.at( headerType );

      // verify compressor
      compressor = getAttributeString( elem, "compressor", "<none>" );
      if( compressor == "<none>" )
         compressor = "";
      if( compressor != "" && compressor != "vtkZLibDataCompressor" )
         throw MeshReaderError( "XMLVTK", "unsupported compressor type: " + compressor + " (only vtkZLibDataCompressor is supported)" );

      // get file type and the corresponding XML element
      fileType = getAttributeString( elem, "type" );
      datasetElement = verifyHasOnlyOneChild( elem, fileType );
#else
      throw_no_tinyxml();
#endif
   }

   virtual VariantVector
   readPointData( std::string arrayName ) override
   {
#ifdef HAVE_TINYXML2
      return readPointOrCellData( "PointData", arrayName );
#else
      throw_no_tinyxml();
#endif
   }

   virtual VariantVector
   readCellData( std::string arrayName ) override
   {
#ifdef HAVE_TINYXML2
      return readPointOrCellData( "CellData", arrayName );
#else
      throw_no_tinyxml();
#endif
   }

   virtual void reset() override
   {
      resetBase();
      fileType = "";
      byteOrder = compressor = headerType = "";
#ifdef HAVE_TINYXML2
      dom.Clear();
      datasetElement = nullptr;
#endif
   }

protected:
   // parsed VTK file type (e.g. "UnstructuredGrid")
   std::string fileType;

   // other attributes parsed from the <VTKFile> element
   std::string byteOrder, compressor, headerType;

#ifdef HAVE_TINYXML2
   // internal attribute representing the whole XML file
   tinyxml2::XMLDocument dom;

   // pointer to the main XML element (e.g. <UnstructuredGrid>)
   const tinyxml2::XMLElement* datasetElement = nullptr;
#endif
};

} // namespace Readers
} // namespace Meshes
} // namespace TNL
