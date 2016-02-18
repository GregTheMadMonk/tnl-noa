/***************************************************************************
                          tnlOperator.h  -  description
                             -------------------
    begin                : Feb 10, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
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


#ifndef TNLOPERATOR_H
#define	TNLOPERATOR_H

#include <functions/tnlDomain.h>

template< typename Mesh,
          tnlDomainType DomainType = MeshInteriorDomain,
          int PreimageEntitiesDimensions = Mesh::getMeshDimensions(),
          int ImageEntitiesDimensions = Mesh::getMeshDimensions(),
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlOperator : public tnlDomain< Mesh::getMeshDimensions(), DomainType >
{
   public:
      
      typedef Mesh MeshType;
      typedef typename MeshType::RealType MeshRealType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType MeshIndexType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef void ExactOperatorType;
      
      constexpr static int getMeshDimensions() { return MeshType::getMeshDimensions(); }
      constexpr static int getPreimageEntitiesDimensions() { return PreimageEntitiesDimensions; }
      constexpr static int getImageEntitiesDimensions() { return ImageEntitiesDimensions; }
      
      bool refresh( const RealType& time = 0.0 ) { return true; }
      
      bool deepRefresh( const RealType& time = 0.0 ) { return true; }
    
      template< typename MeshFunction > 
      void setPreimageFunction( const MeshFunction& f ){};
};
#endif	/* TNLOPERATOR_H */

