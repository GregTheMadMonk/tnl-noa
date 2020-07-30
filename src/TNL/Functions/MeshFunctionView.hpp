/***************************************************************************
                          MeshFunctionView_impl.h  -  description
                             -------------------
    begin                : Nov 8, 2015
    copyright            : (C) 2015 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Assert.h>
#include <TNL/Pointers/DevicePointer.h>
#include <TNL/Functions/MeshFunctionView.h>
#include <TNL/Functions/MeshFunctionEvaluator.h>
#include <TNL/Functions/MeshFunctionNormGetter.h>
#include <TNL/Functions/MeshFunctionGnuplotWriter.h>
#include <TNL/Meshes/Writers/VTKWriter.h>
#include <TNL/Meshes/Writers/VTUWriter.h>

#pragma once

namespace TNL {
namespace Functions {

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
MeshFunctionView()
{
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
MeshFunctionView( const MeshFunctionView& meshFunction )
{
   this->meshPointer = meshFunction.meshPointer;
   this->data.bind( meshFunction.getData() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Vector >
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
MeshFunctionView( const MeshPointer& meshPointer,
              Vector& data,
              const IndexType& offset )
//: meshPointer( meshPointer )
{
   TNL_ASSERT_GE( data.getSize(), meshPointer->template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >(),
                  "The input vector is not large enough for binding to the mesh function." );

   this->meshPointer = meshPointer;
   this->data.bind( data, offset, getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
}


template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Vector >
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
MeshFunctionView( const MeshPointer& meshPointer,
              Pointers::SharedPointer<  Vector >& data,
              const IndexType& offset )
//: meshPointer( meshPointer )
{
   TNL_ASSERT_GE( data->getSize(), offset + meshPointer->template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >(),
                  "The input vector is not large enough for binding to the mesh function." );

   this->meshPointer = meshPointer;
   this->data.bind( *data, offset, getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
String
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
getSerializationType()
{
   return String( "Functions::MeshFunction< " ) +
          TNL::getSerializationType< Mesh >() + ", " +
          convertToString( MeshEntityDimension ) + ", " +
          getType< Real >() +
          " >";
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
String
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
void
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
   config.addEntry< String >( prefix + "file", "Dataset for the mesh function." );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
bool
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
setup( const MeshPointer& meshPointer,
       const Config::ParameterContainer& parameters,
       const String& prefix )
{
   this->setMesh( meshPointer );
   if( parameters.checkParameter( prefix + "file" ) )
   {
      String fileName = parameters.getParameter< String >( prefix + "file" );
      this->load( fileName );
   }
   else
   {
      throw std::runtime_error( "Missing parameter " + prefix + "file." );
   }
   return true;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
void
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
bind( MeshFunctionView& meshFunction )
{
   this->meshPointer = meshFunction.meshPointer;
   this->data.bind( meshFunction.getData() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Vector >
void
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
bind( Vector& data,
      const IndexType& offset )
{
   TNL_ASSERT_GE( data.getSize(), offset + meshPointer->template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >(),
                  "The input vector is not large enough for binding to the mesh function." );
   this->data.bind( data.getData() + offset, getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Vector >
void
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
bind( const MeshPointer& meshPointer,
      Vector& data,
      const IndexType& offset )
{
   TNL_ASSERT_GE( data.getSize(), offset + meshPointer->template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >(),
                  "The input vector is not large enough for binding to the mesh function." );

   this->meshPointer = meshPointer;
   this->data.bind( data.getData() + offset, getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Vector >
void
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
bind( const MeshPointer& meshPointer,
      Pointers::SharedPointer< Vector >& data,
      const IndexType& offset )
{
   TNL_ASSERT_GE( data->getSize(), offset + meshPointer->template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >(),
                  "The input vector is not large enough for binding to the mesh function." );
   static_assert( std::is_same< typename Vector::RealType, RealType >::value, "Cannot bind Vector with different Real type." );

   this->meshPointer = meshPointer;
   this->data.bind( *data + offset, getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
void
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
setMesh( const MeshPointer& meshPointer )
{
   this->meshPointer = meshPointer;
   this->data.reset();
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
 template< typename Device >
__cuda_callable__
const typename MeshFunctionView< Mesh, MeshEntityDimension, Real >::MeshType& 
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
getMesh() const
{
   return this->meshPointer.template getData< Device >();
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
const typename MeshFunctionView< Mesh, MeshEntityDimension, Real >::MeshPointer&
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
getMeshPointer() const
{
   return this->meshPointer;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
typename MeshFunctionView< Mesh, MeshEntityDimension, Real >::IndexType
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
getDofs( const MeshPointer& meshPointer )
{
   return meshPointer->template getEntitiesCount< getEntitiesDimension() >();
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
__cuda_callable__
const typename MeshFunctionView< Mesh, MeshEntityDimension, Real >::VectorType& 
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
getData() const
{
   return this->data;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
__cuda_callable__
typename MeshFunctionView< Mesh, MeshEntityDimension, Real >::VectorType& 
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
getData()
{
   return this->data;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
bool
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
refresh( const RealType& time ) const
{
   return true;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
bool
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
deepRefresh( const RealType& time ) const
{
   return true;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename EntityType >
typename Functions::MeshFunctionView< Mesh, MeshEntityDimension, Real >::RealType
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
getValue( const EntityType& meshEntity ) const
{
   static_assert( EntityType::getEntityDimension() == MeshEntityDimension, "Calling with wrong EntityType -- entity dimensions do not match." );
   return this->data.getElement( meshEntity.getIndex() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename EntityType >
void
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
setValue( const EntityType& meshEntity,
          const RealType& value )
{
   static_assert( EntityType::getEntityDimension() == MeshEntityDimension, "Calling with wrong EntityType -- entity dimensions do not match." );
   this->data.setElement( meshEntity.getIndex(), value );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename EntityType >
__cuda_callable__
typename Functions::MeshFunctionView< Mesh, MeshEntityDimension, Real >::RealType&
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
operator()( const EntityType& meshEntity,
            const RealType& time )
{
   static_assert( EntityType::getEntityDimension() == MeshEntityDimension, "Calling with wrong EntityType -- entity dimensions do not match." );
   return this->data[ meshEntity.getIndex() ];
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename EntityType >
__cuda_callable__
const typename Functions::MeshFunctionView< Mesh, MeshEntityDimension, Real >::RealType&
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
operator()( const EntityType& meshEntity,
            const RealType& time ) const
{
   static_assert( EntityType::getEntityDimension() == MeshEntityDimension, "Calling with wrong EntityType -- entity dimensions do not match." );
   return this->data[ meshEntity.getIndex() ];
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
__cuda_callable__
typename Functions::MeshFunctionView< Mesh, MeshEntityDimension, Real >::RealType&
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
operator[]( const IndexType& meshEntityIndex )
{
   return this->data[ meshEntityIndex ];
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
__cuda_callable__
const typename Functions::MeshFunctionView< Mesh, MeshEntityDimension, Real >::RealType&
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
operator[]( const IndexType& meshEntityIndex ) const
{
   return this->data[ meshEntityIndex ];
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
MeshFunctionView< Mesh, MeshEntityDimension, Real >&
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
operator = ( const MeshFunctionView& f )
{
   this->setMesh( f.getMeshPointer() );
   this->getData() = f.getData();
   return *this;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Function >
MeshFunctionView< Mesh, MeshEntityDimension, Real >&
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
operator = ( const Function& f )
{
   Pointers::DevicePointer< MeshFunctionView > thisDevicePtr( *this );
   Pointers::DevicePointer< std::add_const_t< Function > > fDevicePtr( f );
   MeshFunctionEvaluator< MeshFunctionView, Function >::evaluate( thisDevicePtr, fDevicePtr );
   return *this;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Function >
MeshFunctionView< Mesh, MeshEntityDimension, Real >&
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
operator += ( const Function& f )
{
   Pointers::DevicePointer< MeshFunctionView > thisDevicePtr( *this );
   Pointers::DevicePointer< std::add_const_t< Function > > fDevicePtr( f );
   MeshFunctionEvaluator< MeshFunctionView, Function >::evaluate( thisDevicePtr, fDevicePtr, ( RealType ) 1.0, ( RealType ) 1.0 );
   return *this;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Function >
MeshFunctionView< Mesh, MeshEntityDimension, Real >&
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
operator -= ( const Function& f )
{
   Pointers::DevicePointer< MeshFunctionView > thisDevicePtr( *this );
   Pointers::DevicePointer< std::add_const_t< Function > > fDevicePtr( f );
   MeshFunctionEvaluator< MeshFunctionView, Function >::evaluate( thisDevicePtr, fDevicePtr, ( RealType ) 1.0, ( RealType ) -1.0 );
   return *this;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
Real
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
getLpNorm( const RealType& p ) const
{
   return MeshFunctionNormGetter< Mesh >::getNorm( *this, p );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
Real
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
getMaxNorm() const
{
   return max( abs( this->data ) );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
void
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
save( File& file ) const
{
   TNL_ASSERT_EQ( this->data.getSize(), this->getMesh().template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >(),
                  "Size of the mesh function data does not match the mesh." );
   Object::save( file );
   file << this->data;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
void
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
load( File& file )
{
   Object::load( file );
   file >> this->data;
   const IndexType meshSize = this->getMesh().template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >();
   if( this->data.getSize() != meshSize )
      throw Exceptions::FileDeserializationError( file.getFileName(), "mesh function data size does not match the mesh size (expected " + std::to_string(meshSize) + ", got " + std::to_string(this->data.getSize()) + ")." );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
void
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
boundLoad( File& file )
{
   Object::load( file );
   file >> this->data.getView();
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
void
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
boundLoad( const String& fileName )
{
   File file;
   file.open( fileName, std::ios_base::in );
   this->boundLoad( file );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
bool
MeshFunctionView< Mesh, MeshEntityDimension, Real >::
write( const String& fileName,
       const String& format ) const
{
   std::fstream file;
   file.open( fileName.getString(), std::ios::out );
   if( ! file )
   {
      std::cerr << "Unable to open a file " << fileName << "." << std::endl;
      return false;
   }
   if( format == "vtk" ) {
      Meshes::Writers::VTKWriter< Mesh > writer( file );
      writer.template writeEntities< getEntitiesDimension() >( *meshPointer );
      if( MeshFunctionView::getEntitiesDimension() == 0 )
         writer.writePointData( getData(), "cellFunctionValues", 1 );
      else
         writer.writeCellData( getData(), "pointFunctionValues", 1 );
   }
   else if( format == "vtu" ) {
      Meshes::Writers::VTUWriter< Mesh > writer( file );
      writer.template writeEntities< getEntitiesDimension() >( *meshPointer );
      if( MeshFunctionView::getEntitiesDimension() == 0 )
         writer.writePointData( getData(), "cellFunctionValues", 1 );
      else
         writer.writeCellData( getData(), "pointFunctionValues", 1 );
   }
   else if( format == "gnuplot" )
      return MeshFunctionGnuplotWriter< MeshFunctionView >::write( *this, file );
   else {
      std::cerr << "Unknown output format: " << format << std::endl;
      return false;
   }
   return true;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
std::ostream&
operator << ( std::ostream& str, const MeshFunctionView< Mesh, MeshEntityDimension, Real >& f )
{
   str << f.getData();
   return str;
}

} // namespace Functions
} // namespace TNL
