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
#include <functions/tnlDomain.h>
#include <functions/tnlMeshFunctionGnuplotWriter.h>
#include <functions/tnlMeshFunctionVTKWriter.h>
#include <core/tnlSharedPointer.h>

#ifndef TNLMESHFUNCTION_H
#define TNLMESHFUNCTION_H

template< typename Mesh,
          int MeshEntityDimensions = Mesh::meshDimensions,
          typename Real = typename Mesh::RealType >
class tnlMeshFunction : 
   public tnlObject,
   public tnlDomain< Mesh::meshDimensions, MeshDomain >
{
   //static_assert( Mesh::DeviceType::DeviceType == Vector::DeviceType::DeviceType,
   //               "Both mesh and vector of a mesh function must reside on the same device.");
   public:
      
      typedef Mesh MeshType;
      typedef tnlSharedPointer< MeshType > MeshPointer;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType IndexType;
      typedef Real RealType;
      typedef tnlVector< RealType, DeviceType, IndexType > VectorType;
      typedef tnlMeshFunction< Mesh, MeshEntityDimensions, Real > ThisType;
      
      static constexpr int getEntitiesDimensions() { return MeshEntityDimensions; };
      
      tnlMeshFunction();
      
      tnlMeshFunction( const MeshPointer& meshPointer );      
      
      tnlMeshFunction( const ThisType& meshFunction );
      
      template< typename Vector >
      tnlMeshFunction( const MeshPointer& meshPointer,
                       Vector& data,
                       const IndexType& offset = 0 );
      
      static tnlString getType();
      
      tnlString getTypeVirtual() const;
      
      static tnlString getSerializationType();

      virtual tnlString getSerializationTypeVirtual() const;      
      
      static void configSetup( tnlConfigDescription& config,
                               const tnlString& prefix = "" );

      bool setup( const tnlParameterContainer& parameters,
                  const tnlString& prefix = "" );      
      
      template< typename Vector >
      void bind( const MeshPointer& meshPointer,
                 const Vector& data,
                 const IndexType& offset = 0 );
      
      void setMesh( const MeshPointer& meshPointer );
      
      template< typename Device = tnlHost >
      __cuda_callable__
      const MeshType& getMesh() const;
      
      const MeshPointer& getMeshPointer() const;
      
      __cuda_callable__ const VectorType& getData() const;      
      
      __cuda_callable__ VectorType& getData();
      
      bool refresh( const RealType& time = 0.0 ) const;
      
      bool deepRefresh( const RealType& time = 0.0 ) const;
      
      template< typename EntityType >
      RealType getValue( const EntityType& meshEntity ) const;
      
      template< typename EntityType >
      void setValue( const EntityType& meshEntity,
                     const RealType& value );
      
      template< typename EntityType >
      __cuda_callable__
      RealType& operator()( const EntityType& meshEntity,
                            const RealType& time = 0.0 );
      
      template< typename EntityType >
      __cuda_callable__
      const RealType& operator()( const EntityType& meshEntity,
                                  const RealType& time = 0.0 ) const;
      
      __cuda_callable__
      RealType& operator[]( const IndexType& meshEntityIndex );
      
      __cuda_callable__
      const RealType& operator[]( const IndexType& meshEntityIndex ) const;

      template< typename Function >
      ThisType& operator = ( const Function& f );
      
      template< typename Function >
      ThisType& operator -= ( const Function& f );

      template< typename Function >
      ThisType& operator += ( const Function& f );
      
      RealType getLpNorm( const RealType& p ) const;
      
      RealType getMaxNorm() const;
      
      bool save( tnlFile& file ) const;

      bool load( tnlFile& file );
      
      bool boundLoad( tnlFile& file );
      
      bool write( const tnlString& fileName,
                  const tnlString& format = "vtk" ) const;
      
      using tnlObject::save;
      
      using tnlObject::load;
            
      using tnlObject::boundLoad;
            
   protected:
      
      MeshPointer meshPointer;
      
      VectorType data;
      
      template< typename, typename > friend class tnlMeshFunctionEvaluator;
};

#include <functions/tnlMeshFunction_impl.h>
#include <functions/tnlMeshFunctionGnuplotWriter_impl.h>
#include <functions/tnlMeshFunctionVTKWriter_impl.h>


#endif	/* TNLMESHFUNCTION_H */

