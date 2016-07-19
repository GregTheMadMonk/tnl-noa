/***************************************************************************
                          navierStokesSetter.h  -  description
                             -------------------
    begin                : Feb 23, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef SIMPLEPROBLEMTYPESSETTER_H_
#define SIMPLEPROBLEMTYPESSETTER_H_

#include <TNL/config/tnlParameterContainer.h>
#include "navierStokesSolver.h"

template< typename MeshType,
          typename SolverStarter >
class navierStokesSetter
{
   public:

   template< typename RealType,
             typename DeviceType,
             typename IndexType >
   static bool run( const tnlParameterContainer& parameters );
};

template< typename MeshReal, typename Device, typename MeshIndex, typename SolverStarter >
class navierStokesSetter< tnlGrid< 2, MeshReal, Device, MeshIndex >, SolverStarter >
{
   public:

   typedef tnlGrid< 2, MeshReal, Device, MeshIndex > MeshType;

   template< typename RealType,
             typename DeviceType,
             typename IndexType >
   static bool run( const tnlParameterContainer& parameters );
};


#include "navierStokesSetter_impl.h"


#endif /* SIMPLEPROBLEMSETTER_H_ */
