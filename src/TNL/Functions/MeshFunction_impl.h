/***************************************************************************
                          MeshFunction_impl.h  -  description
                             -------------------
    begin                : Nov 8, 2015
    copyright            : (C) 2015 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Assert.h>
#include <TNL/Pointers/DevicePointer.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Functions/MeshFunctionEvaluator.h>
#include <TNL/Functions/MeshFunctionNormGetter.h>
#include <TNL/Functions/MeshFunctionGnuplotWriter.h>
#include <TNL/Functions/MeshFunctionVTKWriter.h>

#pragma once

namespace TNL {
namespace Functions {

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
MeshFunction< Mesh, MeshEntityDimension, Real >::
MeshFunction()
{
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
MeshFunction< Mesh, MeshEntityDimension, Real >::
MeshFunction( const MeshPointer& meshPointer )
{

    setupSynchronizer(meshPointer->getDistributedMesh());

   this->meshPointer=meshPointer;
   this->data.setSize( getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
MeshFunction< Mesh, MeshEntityDimension, Real >::
MeshFunction( const MeshFunction& meshFunction )
{
    setupSynchronizer(meshFunction.meshPointer->getDistributedMesh());

   this->meshPointer=meshFunction.meshPointer;
   this->data.bind( meshFunction.getData() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Vector >
MeshFunction< Mesh, MeshEntityDimension, Real >::
MeshFunction( const MeshPointer& meshPointer,
              Vector& data,
              const IndexType& offset )
//: meshPointer( meshPointer )
{
   TNL_ASSERT_GE( data.getSize(), meshPointer->template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >(),
                  "The input vector is not large enough for binding to the mesh function." );
    setupSynchronizer(meshPointer->getDistributedMesh());

   this->meshPointer=meshPointer;
   this->data.bind( data, offset, getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
}


template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Vector >
MeshFunction< Mesh, MeshEntityDimension, Real >::
MeshFunction( const MeshPointer& meshPointer,
              Pointers::SharedPointer<  Vector >& data,
              const IndexType& offset )
//: meshPointer( meshPointer )
{
   TNL_ASSERT_GE( data->getSize(), offset + meshPointer->template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >(),
                  "The input vector is not large enough for binding to the mesh function." );

    setupSynchronizer(meshPointer->getDistributedMesh());

   this->meshPointer=meshPointer;
   this->data.bind( *data, offset, getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
String
MeshFunction< Mesh, MeshEntityDimension, Real >::
getType()
{
   return String( "Functions::MeshFunction< " ) +
                     Mesh::getType() + ", " +
                     convertToString( MeshEntityDimension ) + ", " +
                    TNL::getType< Real >() +
                     " >";
};

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
String
MeshFunction< Mesh, MeshEntityDimension, Real >::
getTypeVirtual() const
{
   return this->getType();
};

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
String
MeshFunction< Mesh, MeshEntityDimension, Real >::
getSerializationType()
{
   return String( "Functions::MeshFunction< " ) +
                     Mesh::getSerializationType() + ", " +
                     convertToString( MeshEntityDimension ) + ", " +
                    TNL::getType< Real >() +
                     " >";
};

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
String
MeshFunction< Mesh, MeshEntityDimension, Real >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
void
MeshFunction< Mesh, MeshEntityDimension, Real >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
   config.addEntry< String >( prefix + "file", "Dataset for the mesh function." );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
bool
MeshFunction< Mesh, MeshEntityDimension, Real >::
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
MeshFunction< Mesh, MeshEntityDimension, Real >::
bind( MeshFunction& meshFunction )
{

    setupSynchronizer(meshFunction.meshPointer->getDistributedMesh());

   this->meshPointer=meshFunction.meshPointer;
   this->data.bind( meshFunction.getData() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Vector >
void
MeshFunction< Mesh, MeshEntityDimension, Real >::
bind( const Vector& data,
      const IndexType& offset )
{
   TNL_ASSERT_GE( data.getSize(), offset + meshPointer->template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >(),
                  "The input vector is not large enough for binding to the mesh function." );
   this->data.bind( data, offset, getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Vector >
void
MeshFunction< Mesh, MeshEntityDimension, Real >::
bind( const MeshPointer& meshPointer,
      const Vector& data,
      const IndexType& offset )
{
   TNL_ASSERT_GE( data.getSize(), offset + meshPointer->template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >(),
                  "The input vector is not large enough for binding to the mesh function." );

   setupSynchronizer(meshPointer->getDistributedMesh());
   this->meshPointer=meshPointer;
   this->data.bind( data, offset, getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Vector >
void
MeshFunction< Mesh, MeshEntityDimension, Real >::
bind( const MeshPointer& meshPointer,
      const Pointers::SharedPointer<  Vector >& data,
      const IndexType& offset )
{
   TNL_ASSERT_GE( data->getSize(), offset + meshPointer->template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >(),
                  "The input vector is not large enough for binding to the mesh function." );

   setupSynchronizer(meshPointer->getDistributedMesh());
   this->meshPointer=meshPointer;
   this->data.bind( *data, offset, getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
void
MeshFunction< Mesh, MeshEntityDimension, Real >::
setMesh( const MeshPointer& meshPointer )
{

   setupSynchronizer(meshPointer->getDistributedMesh());

   this->meshPointer=meshPointer;
   this->data.setSize( getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
 template< typename Device >
__cuda_callable__
const typename MeshFunction< Mesh, MeshEntityDimension, Real >::MeshType& 
MeshFunction< Mesh, MeshEntityDimension, Real >::
getMesh() const
{
   return this->meshPointer.template getData< Device >();
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
const typename MeshFunction< Mesh, MeshEntityDimension, Real >::MeshPointer&
MeshFunction< Mesh, MeshEntityDimension, Real >::
getMeshPointer() const
{
   return this->meshPointer;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
typename MeshFunction< Mesh, MeshEntityDimension, Real >::IndexType
MeshFunction< Mesh, MeshEntityDimension, Real >::
getDofs( const MeshPointer& meshPointer )
{
   return meshPointer->template getEntitiesCount< getEntitiesDimension() >();
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
__cuda_callable__
const typename MeshFunction< Mesh, MeshEntityDimension, Real >::VectorType& 
MeshFunction< Mesh, MeshEntityDimension, Real >::
getData() const
{
   return this->data;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
__cuda_callable__
typename MeshFunction< Mesh, MeshEntityDimension, Real >::VectorType& 
MeshFunction< Mesh, MeshEntityDimension, Real >::
getData()
{
   return this->data;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
bool
MeshFunction< Mesh, MeshEntityDimension, Real >::
refresh( const RealType& time ) const
{
   return true;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
bool
MeshFunction< Mesh, MeshEntityDimension, Real >::
deepRefresh( const RealType& time ) const
{
   return true;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename EntityType >
typename Functions::MeshFunction< Mesh, MeshEntityDimension, Real >::RealType
MeshFunction< Mesh, MeshEntityDimension, Real >::
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
MeshFunction< Mesh, MeshEntityDimension, Real >::
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
typename Functions::MeshFunction< Mesh, MeshEntityDimension, Real >::RealType&
MeshFunction< Mesh, MeshEntityDimension, Real >::
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
const typename Functions::MeshFunction< Mesh, MeshEntityDimension, Real >::RealType&
MeshFunction< Mesh, MeshEntityDimension, Real >::
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
typename Functions::MeshFunction< Mesh, MeshEntityDimension, Real >::RealType&
MeshFunction< Mesh, MeshEntityDimension, Real >::
operator[]( const IndexType& meshEntityIndex )
{
   return this->data[ meshEntityIndex ];
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
__cuda_callable__
const typename Functions::MeshFunction< Mesh, MeshEntityDimension, Real >::RealType&
MeshFunction< Mesh, MeshEntityDimension, Real >::
operator[]( const IndexType& meshEntityIndex ) const
{
   return this->data[ meshEntityIndex ];
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
MeshFunction< Mesh, MeshEntityDimension, Real >&
MeshFunction< Mesh, MeshEntityDimension, Real >::
operator = ( const MeshFunction& f )
{
   this->setMesh( f.getMeshPointer() );
   this->getData() = f.getData();
   return *this;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Function >
MeshFunction< Mesh, MeshEntityDimension, Real >&
MeshFunction< Mesh, MeshEntityDimension, Real >::
operator = ( const Function& f )
{
   Pointers::DevicePointer< MeshFunction > thisDevicePtr( *this );
   Pointers::DevicePointer< std::add_const_t< Function > > fDevicePtr( f );
   MeshFunctionEvaluator< MeshFunction, Function >::evaluate( thisDevicePtr, fDevicePtr );
   return *this;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Function >
MeshFunction< Mesh, MeshEntityDimension, Real >&
MeshFunction< Mesh, MeshEntityDimension, Real >::
operator += ( const Function& f )
{
   Pointers::DevicePointer< MeshFunction > thisDevicePtr( *this );
   Pointers::DevicePointer< std::add_const_t< Function > > fDevicePtr( f );
   MeshFunctionEvaluator< MeshFunction, Function >::evaluate( thisDevicePtr, fDevicePtr, ( RealType ) 1.0, ( RealType ) 1.0 );
   return *this;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Function >
MeshFunction< Mesh, MeshEntityDimension, Real >&
MeshFunction< Mesh, MeshEntityDimension, Real >::
operator -= ( const Function& f )
{
   Pointers::DevicePointer< MeshFunction > thisDevicePtr( *this );
   Pointers::DevicePointer< std::add_const_t< Function > > fDevicePtr( f );
   MeshFunctionEvaluator< MeshFunction, Function >::evaluate( thisDevicePtr, fDevicePtr, ( RealType ) 1.0, ( RealType ) -1.0 );
   return *this;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
Real
MeshFunction< Mesh, MeshEntityDimension, Real >::
getLpNorm( const RealType& p ) const
{
   return MeshFunctionNormGetter< MeshFunction >::getNorm( *this, p );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
Real
MeshFunction< Mesh, MeshEntityDimension, Real >::
getMaxNorm() const
{
   return max( abs( this->data ) );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
void
MeshFunction< Mesh, MeshEntityDimension, Real >::
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
MeshFunction< Mesh, MeshEntityDimension, Real >::
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
MeshFunction< Mesh, MeshEntityDimension, Real >::
boundLoad( File& file )
{
   Object::load( file );
   file >> this->data.getView();
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
void
MeshFunction< Mesh, MeshEntityDimension, Real >::
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
MeshFunction< Mesh, MeshEntityDimension, Real >::
write( const String& fileName,
       const String& format,
       const double& scale ) const
{
   std::fstream file;
   file.open( fileName.getString(), std::ios::out );
   if( ! file )
   {
      std::cerr << "Unable to open a file " << fileName << "." << std::endl;
      return false;
   }
   if( format == "vtk" )
      return MeshFunctionVTKWriter< MeshFunction >::write( *this, file, scale );
   else if( format == "gnuplot" )
      return MeshFunctionGnuplotWriter< MeshFunction >::write( *this, file, scale );
   else {
      std::cerr << "Unknown output format: " << format << std::endl;
      return false;
   }
   return true;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
template< typename CommunicatorType,
          typename PeriodicBoundariesMaskType >
void
MeshFunction< Mesh, MeshEntityDimension, Real >:: 
synchronize( bool periodicBoundaries,
             const Pointers::SharedPointer< PeriodicBoundariesMaskType, DeviceType >& mask )
{
    auto distrMesh = this->getMesh().getDistributedMesh();
    if(distrMesh != NULL && distrMesh->isDistributed())
    {
        this->synchronizer.template synchronize<CommunicatorType>( *this, periodicBoundaries, mask );
    }
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
void
MeshFunction< Mesh, MeshEntityDimension, Real >:: 
setupSynchronizer( DistributedMeshType *distributedMesh )
{
   if( distributedMesh )
      this->synchronizer.setDistributedGrid( distributedMesh );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
std::ostream&
operator << ( std::ostream& str, const MeshFunction< Mesh, MeshEntityDimension, Real >& f )
{
   str << f.getData();
   return str;
}

} // namespace Functions
} // namespace TNL
