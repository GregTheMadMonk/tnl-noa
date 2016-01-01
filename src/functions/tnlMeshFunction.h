/***************************************************************************
                          tnlMeshFunction.h  -  description
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

#include <core/tnlObject.h>
#include <functions/tnlFunction.h>

#ifndef TNLMESHFUNCTION_H
#define TNLMESHFUNCTION_H

template< typename Mesh,
          int MeshEntityDimensions = Mesh::meshDimensions,
          typename Real = typename Mesh::RealType >
class tnlMeshFunction : 
   public tnlObject,
   public tnlFunction< Mesh::meshDimensions, MeshFunction >
{
   //static_assert( Mesh::DeviceType::DeviceType == Vector::DeviceType::DeviceType,
   //               "Both mesh and vector of a mesh function must reside on the same device.");
   public:
      
      typedef Mesh MeshType;      
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType IndexType;
      typedef Real RealType;
      typedef tnlVector< RealType, DeviceType, IndexType > VectorType;
      
      static constexpr int getMeshEntityDimensions() { return  MeshEntityDimensions; };
      
      tnlMeshFunction();
      
      tnlMeshFunction( const MeshType& mesh );
      
      template< typename Vector >
      tnlMeshFunction( const MeshType& mesh,
                       Vector& data,
                       const IndexType& offset = 0 );
      
      static void configSetup( tnlConfigDescription& config,
                               const tnlString& prefix = "" );

      bool setup( const tnlParameterContainer& parameters,
                  const tnlString& prefix = "" );      
      
      template< typename Vector >
      void bind( const MeshType& mesh,
                 const Vector& data,
                 const IndexType& offset = 0 );
      
      void setMesh( const MeshType& mesh ) const;      
      
      const MeshType& getMesh() const;      
      
      const VectorType& getData() const;      
      
      VectorType& getData();      
      
      template< typename EntityType >
      RealType getValue( const EntityType& meshEntity ) const;
      
      template< typename EntityType >
      void setValue( const EntityType& meshEntity,
                     const RealType& value );
      
      template< typename EntityType >
      __cuda_callable__
      RealType& operator()( const EntityType& meshEntity );
      
      template< typename EntityType >
      __cuda_callable__
      const RealType& operator()( const EntityType& meshEntity ) const;
      
      __cuda_callable__
      RealType& operator[]( const IndexType& meshEntityIndex );
      
      __cuda_callable__
      const RealType& operator[]( const IndexType& meshEntityIndex ) const;

      
      bool save( tnlFile& file ) const;

      bool load( tnlFile& file );
      
      using tnlObject::load;

      using tnlObject::save;
            
   protected:
      
      const MeshType* mesh;
      
      VectorType data;     
};

#include <functions/tnlMeshFunction_impl.h>

#endif	/* TNLMESHFUNCTION_H */

