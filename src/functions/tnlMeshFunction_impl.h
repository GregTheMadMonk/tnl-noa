/***************************************************************************
                          tnlMeshFunction_impl.h  -  description
                             -------------------
    begin                : Nov 8, 2015
    copyright            : (C) 2015 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include <core/tnlAssert.h>
#include <functions/tnlMeshFunction.h>
#include <functions/tnlFunctionEvaluator.h>
#include <functions/tnlMeshFunctionEvaluator.h>
#include <functions/tnlMeshFunctionNormGetter.h>
#include <functions/tnlMeshFunctionGnuplotWriter.h>
#include <functions/tnlMeshFunctionVTKWriter.h>

#ifndef TNLMESHFUNCTION_IMPL_H
#define	TNLMESHFUNCTION_IMPL_H

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
tnlString 
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
getType()
{
   return tnlString( "tnlMeshFunction< " ) +
                     Mesh::getType() + ", " +
                     tnlString( MeshEntityDimensions ) + ", " +
                     ::getType< Real >() +
                     " >";
};

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
tnlString 
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
getTypeVirtual() const
{
   return this->getType();
};

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
tnlString 
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
getSerializationType()
{
   return tnlString( "tnlMeshFunction< " ) +
                     Mesh::getSerializationType() + ", " +
                     tnlString( MeshEntityDimensions ) + ", " +
                     ::getType< Real >() +
                     " >";
};

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
tnlString 
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
configSetup( tnlConfigDescription& config,
             const tnlString& prefix )
{
   config.addEntry< tnlString >( prefix + "file", "Dataset for the mesh function." );   
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
bool
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
setup( const tnlParameterContainer& parameters,
       const tnlString& prefix )
{
   if( parameters.checkParameter( prefix + "file" ) )
   {
      tnlString fileName = parameters.getParameter< tnlString >( prefix + "file" );
      if( ! this->data.load( fileName ) )
         return false;
   }
   return true;
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
setMesh( const MeshType& mesh ) const
{
   this->mesh = &mesh;
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
const typename tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::MeshType& 
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
getMesh() const
{
   return *this->mesh;
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
const typename tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::VectorType& 
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
getData() const
{
   return this->data;
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
typename tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::VectorType& 
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
getData()
{
   return this->data;
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
   template< typename EntityType >          
typename tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::RealType
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
typename tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::RealType& 
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
const typename tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::RealType& 
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
typename tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::RealType& 
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
operator[]( const IndexType& meshEntityIndex )
{   
   return this->data[ meshEntityIndex ];
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
__cuda_callable__
const typename tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::RealType& 
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
save( tnlFile& file ) const
{
   if( ! tnlObject::save( file ) )
      return false;
   return this->data.save( file );
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
bool
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
load( tnlFile& file )
{
   if( ! tnlObject::load( file ) )
      return false;
   return this->data.load( file );   
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
bool
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
boundLoad( tnlFile& file )
{
   if( ! tnlObject::load( file ) )
      return false;
   return this->data.boundLoad( file );   
}

template< typename Mesh,
          int MeshEntityDimensions,
          typename Real >
bool
tnlMeshFunction< Mesh, MeshEntityDimensions, Real >::
write( const tnlString& fileName,
       const tnlString& format ) const
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
      


#endif	/* TNLMESHFUNCTION_IMPL_H */

