/***************************************************************************
                          tnlMeshFunction_impl.h  -  description
                             -------------------
    begin                : Nov 8, 2015
    copyright            : (C) 2015 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Assert.h>
#include <TNL/Functions/tnlMeshFunction.h>
#include <TNL/Functions/tnlFunctionEvaluator.h>
#include <TNL/Functions/tnlMeshFunctionEvaluator.h>
#include <TNL/Functions/tnlMeshFunctionNormGetter.h>
#include <TNL/Functions/tnlMeshFunctionGnuplotWriter.h>
#include <TNL/Functions/tnlMeshFunctionVTKWriter.h>

#pragma once

namespace TNL {
namespace Functions {   

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
tnlMeshFunction()
: mesh( 0 )
{
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
tnlMeshFunction( const Mesh& mesh )
: mesh( &mesh )
{
   this->data.setSize( mesh.template getEntitiesCount< typename Mesh::template MeshEntity< MeshEntityDimensions > >() );
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
   template< typename Vector >
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
tnlMeshFunction( const MeshType& mesh,
                 Vector& data,
                 const IndexType& offset )
{
   this->bind( mesh, data, offset );
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
String
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
getType()
{
   return String( "tnlMeshFunction< " ) +
                     Mesh::getType() + ", " +
                     String( MeshEntityDimensions ) + ", " +
                    TNL::getType< Real >() +
                     " >";
};

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
String
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
getTypeVirtual() const
{
   return this->getType();
};

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
String
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
getSerializationType()
{
   return String( "tnlMeshFunction< " ) +
                     Mesh::getSerializationType() + ", " +
                     String( MeshEntityDimensions ) + ", " +
                    TNL::getType< Real >() +
                     " >";
};

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
String
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
void
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
   config.addEntry< String >( prefix + "file", "Dataset for the mesh function." );
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
bool
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   if( parameters.checkParameter( prefix + "file" ) )
   {
      String fileName = parameters.getParameter< String >( prefix + "file" );
      if( ! this->data.load( fileName ) )
         return false;
   }
   return true;
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
void
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
bind( tnlMeshFunction< Mesh, MeshEntityDimensions, Real >& meshFunction )
{
   this->mesh = &meshFunction.getMesh();
   this->data.bind( meshFunction.getData() );
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
   template< typename Vector >
void
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
bind( const MeshType& mesh,
      const Vector& data,
      const IndexType& offset )
{
   this->mesh = &mesh;
   this->data.bind( data, offset, mesh.template getEntitiesCount< typename Mesh::template MeshEntity< MeshEntityDimensions > >() );
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
void
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
setMesh( const MeshType& mesh )
{
   this->mesh = &mesh;
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
const typename Functions::tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::MeshType&
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
getMesh() const
{
   return *this->mesh;
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
const typename Functions::tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::VectorType&
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
getData() const
{
   return this->data;
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
typename Functions::tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::VectorType&
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
getData()
{
   return this->data;
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
bool
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
refresh( const RealType& time ) const
{
   return true;
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
bool
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
deepRefresh( const RealType& time ) const
{
   return true;
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
   template< typename EntityType >
typename Functions::tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::RealType
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
getValue( const EntityType& meshEntity ) const
{
   static_assert( EntityType::entityDimensions == MeshEntityDimensions, "Calling with wrong EntityType -- entity dimensions do not match." );
   return this->data.getValue( meshEntity.getIndex() );
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
   template< typename EntityType >
void
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
setValue( const EntityType& meshEntity,
          const RealType& value )
{
   static_assert( EntityType::entityDimensions == MeshEntityDimensions, "Calling with wrong EntityType -- entity dimensions do not match." );
   this->data.setValue( meshEntity.getIndex(), value );
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
   template< typename EntityType >
__cuda_callable__
typename Functions::tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::RealType&
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
operator()( const EntityType& meshEntity,
            const RealType& time )
{
   static_assert( EntityType::entityDimensions == MeshEntityDimensions, "Calling with wrong EntityType -- entity dimensions do not match." );
   return this->data[ meshEntity.getIndex() ];
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
   template< typename EntityType >
__cuda_callable__
const typename Functions::tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::RealType&
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
operator()( const EntityType& meshEntity,
            const RealType& time ) const
{
   static_assert( EntityType::entityDimensions == MeshEntityDimensions, "Calling with wrong EntityType -- entity dimensions do not match." );
   return this->data[ meshEntity.getIndex() ];
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
__cuda_callable__
typename Functions::tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::RealType&
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
operator[]( const IndexType& meshEntityIndex )
{
   return this->data[ meshEntityIndex ];
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
__cuda_callable__
const typename Functions::tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::RealType&
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
operator[]( const IndexType& meshEntityIndex ) const
{
   return this->data[ meshEntityIndex ];
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
   template< typename Function >
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >&
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
operator = ( const Function& f )
{
   tnlMeshFunctionEvaluator< ThisType, Function >::evaluate( *this, f );
   return *this;
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
   template< typename Function >
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >&
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
operator += ( const Function& f )
{
   tnlMeshFunctionEvaluator< ThisType, Function >::evaluate( *this, f, 1.0, 1.0 );
   return *this;
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
   template< typename Function >
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >&
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
operator -= ( const Function& f )
{
   tnlMeshFunctionEvaluator< ThisType, Function >::evaluate( *this, f, 1.0, -1.0 );
   return *this;
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
Real
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
getLpNorm( const RealType& p ) const
{
   return tnlMeshFunctionNormGetter< ThisType >::getNorm( *this, p );
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
Real
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
getMaxNorm() const
{
   return this->data.absMax();
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
bool
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
save( File& file ) const
{
   if( ! Object::save( file ) )
      return false;
   return this->data.save( file );
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
bool
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
load( File& file )
{
   if( ! Object::load( file ) )
      return false;
   return this->data.load( file );
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
bool
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
boundLoad( File& file )
{
   if( ! Object::load( file ) )
      return false;
   return this->data.boundLoad( file );
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
bool
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
write( const String& fileName,
       const String& format ) const
{
   std::fstream file;
   file.open( fileName.getString(), std::ios::out );
   if( ! file )
   {
      std::cerr << "Unbable to open a file " << fileName << "." << std::endl;
      return false;
   }
   if( format == "vtk" )
      return tnlMeshFunctionVTKWriter< ThisType >::write( *this, file );
   if( format == "gnuplot" )
      return tnlMeshFunctionGnuplotWriter< ThisType >::write( *this, file );
   return true;
}
 
} // namespace Functions
} // namespace TNL

