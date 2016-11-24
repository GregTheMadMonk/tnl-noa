/***************************************************************************
                          MeshFunction_impl.h  -  description
                             -------------------
    begin                : Nov 8, 2015
    copyright            : (C) 2015 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Assert.h>
#include <TNL/DevicePointer.h>
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
: meshPointer( meshPointer )
{
   this->data.setSize( getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
   TNL_ASSERT( this->data.getSize() == this->getMesh().template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >(), 
               std::cerr << "this->data.getSize() = " << this->data.getSize() << std::endl
                         << "this->getMesh().template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >() = " << this->getMesh().template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >() );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
MeshFunction< Mesh, MeshEntityDimension, Real >::
MeshFunction( const ThisType& meshFunction )
: meshPointer( meshFunction.meshPointer )
{
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
: meshPointer( meshPointer )
{
   this->data.bind( data, offset, getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
   TNL_ASSERT( this->data.getSize() == this->getMesh().template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >(), 
               std::cerr << "this->data.getSize() = " << this->data.getSize() << std::endl
                         << "this->getMesh().template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >() = " << this->getMesh().template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >() );   
}


template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Vector >
MeshFunction< Mesh, MeshEntityDimension, Real >::
MeshFunction( const MeshPointer& meshPointer,
              SharedPointer< Vector >& data,
              const IndexType& offset )
: meshPointer( meshPointer )
{
   this->data.bind( *data, offset, getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
   TNL_ASSERT( this->data.getSize() == this->getMesh().template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >(), 
               std::cerr << "this->data.getSize() = " << this->data.getSize() << std::endl
                         << "this->getMesh().template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >() = " << this->getMesh().template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >() );   
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
                     String( MeshEntityDimension ) + ", " +
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
                     String( MeshEntityDimension ) + ", " +
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
      if( ! this->load( fileName ) )
         return false;
   }
   else
   {
      std::cerr << "Missing parameter " << prefix << "file." << std::endl;
      throw(0);
      return false;
   }
   return true;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
void
MeshFunction< Mesh, MeshEntityDimension, Real >::
bind( ThisType& meshFunction )
{
   this->meshPointer = meshFunction.getMeshPointer();
   this->data.bind( meshFunction.getData() );
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
   this->meshPointer = meshPointer;
   this->data.bind( data, offset, getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
   TNL_ASSERT( this->data.getSize() == this->getMesh().template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >(), 
               std::cerr << "this->data.getSize() = " << this->data.getSize() << std::endl
                         << "this->getMesh().template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >() = " << this->getMesh().template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >() );   
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
   template< typename Vector >
void
MeshFunction< Mesh, MeshEntityDimension, Real >::
bind( const MeshPointer& meshPointer,
      const SharedPointer< Vector >& data,
      const IndexType& offset )
{
   this->meshPointer = meshPointer;
   this->data.bind( *data, offset, getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
   TNL_ASSERT( this->data.getSize() == this->getMesh().template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >(), 
               std::cerr << "this->data.getSize() = " << this->data.getSize() << std::endl
                         << "this->getMesh().template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >() = " << this->getMesh().template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >() );   
}


template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
void
MeshFunction< Mesh, MeshEntityDimension, Real >::
setMesh( const MeshPointer& meshPointer )
{
   this->meshPointer = meshPointer;
   this->data.setSize( getMesh().template getEntitiesCount< typename Mesh::template EntityType< MeshEntityDimension > >() );
   TNL_ASSERT( this->data.getSize() == this->getMesh().template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >(), 
               std::cerr << "this->data.getSize() = " << this->data.getSize() << std::endl
                         << "this->getMesh().template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >() = " << this->getMesh().template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >() );   
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
__cuda_callable__
typename MeshFunction< Mesh, MeshEntityDimension, Real >::IndexType
MeshFunction< Mesh, MeshEntityDimension, Real >::
getDofs( const MeshPointer& meshPointer )
{
   // FIXME: SharedPointer::operator->() is not __cuda_callable__, but SharedPointer::getData() needs a device parameter
   // solution: shared pointers should not be passed to any __cuda_callable__ function, pass the wrapped object directly
   // (i.e. here the getDofs method should take the mesh, not meshPointer)
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
   return this->data.getValue( meshEntity.getIndex() );
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
   this->data.setValue( meshEntity.getIndex(), value );
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
   template< typename Function >
MeshFunction< Mesh, MeshEntityDimension, Real >&
MeshFunction< Mesh, MeshEntityDimension, Real >::
operator = ( const Function& f )
{
   DevicePointer< ThisType > thisDevicePtr( *this );
   DevicePointer< typename std::add_const< Function >::type > fDevicePtr( f );
   MeshFunctionEvaluator< ThisType, Function >::evaluate( thisDevicePtr, fDevicePtr );
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
   DevicePointer< ThisType > thisDevicePtr( *this );
   DevicePointer< typename std::add_const< Function >::type > fDevicePtr( f );
   MeshFunctionEvaluator< ThisType, Function >::evaluate( thisDevicePtr, fDevicePtr, 1.0, 1.0 );
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
   DevicePointer< ThisType > thisDevicePtr( *this );
   DevicePointer< typename std::add_const< Function >::type > fDevicePtr( f );
   MeshFunctionEvaluator< ThisType, Function >::evaluate( thisDevicePtr, fDevicePtr, 1.0, -1.0 );
   return *this;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
Real
MeshFunction< Mesh, MeshEntityDimension, Real >::
getLpNorm( const RealType& p ) const
{
   return MeshFunctionNormGetter< ThisType >::getNorm( *this, p );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
Real
MeshFunction< Mesh, MeshEntityDimension, Real >::
getMaxNorm() const
{
   return this->data.absMax();
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
bool
MeshFunction< Mesh, MeshEntityDimension, Real >::
save( File& file ) const
{
   TNL_ASSERT( this->data.getSize() == this->getMesh().template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >(), 
               std::cerr << "this->data.getSize() = " << this->data.getSize() << std::endl
                         << "this->getMesh().template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >() = " << this->getMesh().template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >() );
   if( ! Object::save( file ) )
      return false;
   return this->data.save( file );
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
bool
MeshFunction< Mesh, MeshEntityDimension, Real >::
load( File& file )
{
   if( ! Object::load( file ) )
      return false;
   if( ! this->data.load( file ) )
      return false;
   const IndexType meshSize = this->getMesh().template getEntitiesCount< typename MeshType::template EntityType< MeshEntityDimension > >();
   if( this->data.getSize() != meshSize )
   {      
      std::cerr << "Size of the data loaded to the mesh function (" << this->data.getSize() << ") does not fit with the mesh size (" << meshSize << ")." << std::endl;
      return false;
   }
   return true;
}

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
bool
MeshFunction< Mesh, MeshEntityDimension, Real >::
boundLoad( File& file )
{
   if( ! Object::load( file ) )
      return false;
   return this->data.boundLoad( file );
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
      return MeshFunctionVTKWriter< ThisType >::write( *this, file, scale );
   else if( format == "gnuplot" )
      return MeshFunctionGnuplotWriter< ThisType >::write( *this, file, scale );
   else {
      std::cerr << "Unknown output format: " << format << std::endl;
      return false;
   }
   return true;
}
 
} // namespace Functions
} // namespace TNL

