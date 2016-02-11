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

template< typename Mesh,
          tnlDomainType DomainType = MeshInteriorDomain,
          int DomainEntitiesDimensions = Mesh::getMeshDimensions(),
          int ImageEntitiesDimensions = Mesh::getMeshDimensions(),
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class tnlOperator : public tnlDomain< Mesh::getMeshDimensions(), DomainType >
{
   public:
      
      typedef Mesh MeshType;
      typedef typename MeshType::RealType RealType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType IndexType;
      
      constexpr static int getDomainEntitiesDimensions() { return DomainEntitiesDimensions; }
      constexpr static int getImageEntitiesDimensions() { return ImageEntitiesDimensions; }
      
      void refresh() {}
};
#endif	/* TNLOPERATOR_H */

