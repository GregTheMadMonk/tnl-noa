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

#include <functors/tnlFunction.h>

#ifndef TNLMESHFUNCTION_H
#define TNLMESHFUNCTION_H

template< typename Mesh,
          typename Vector,
          int MeshEntitiesDimensions_ = Mesh::Dimensions >
class tnlMeshFunction : public tnlFunction< Mesh::Dimensions,
                                            typename VectorType::RealType,
                                            typename VectorType::IndexType >
{
   static_assert( Mesh::DeviceType::DeviceType == Vector::DeviceType::DeviceType,
                  "Both mesh and vector of a mesh function must reside on the same device.");
   public:
      
      typedef Mesh MeshType;      
      typedef Vector VectorType;
      typedef typename VectorType::RealType RealType;
      typedef typename VectorType::DeviceType DeviceType;
      typedef typename VectorType::IndexType IndexType;
      
      static constexpr int MeshEntitiesDimensions = MeshEntitiesDimensions_;
      
      tnlMeshFunction();
      
      tnlMeshFunction( const MeshType* mesh );
      
      void setMesh( const MeshType* mesh );
      
      const MeshType& getMesh() const;
      
      const VectorType& getData() const;
      
      const RealType& getValue( const IndexType meshEntityIndex );
      
      void setValue( const IndexType meshEntityIndex,
                     const RealType& value );
      
      RealType& operator[]( const IndexType meshEntityIndex );
      
      const RealType& operator[]( const IndexType meshEntityIndex ) const;
      
   protected:
      
      const MeshType* mesh;
      
      VectorType data;
      
};


#endif	/* TNLMESHFUNCTION_H */

