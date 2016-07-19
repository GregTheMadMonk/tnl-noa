/***************************************************************************
                          navierStokesSetter_impl.h  -  description
                             -------------------
    begin                : Mar 9, 2013
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

#ifndef NAVIERSTOKESSETTER_IMPL_H_
#define NAVIERSTOKESSETTER_IMPL_H_

#include <TNL/mesh/tnlGrid.h>
#include <TNL/mesh/tnlLinearGridGeometry.h>
#include <TNL/operators/euler/fvm/tnlLaxFridrichs.h>
#include <TNL/operators/gradient/tnlCentralFDMGradient.h>

template< typename MeshType, typename SolverStarter >
   template< typename RealType,
             typename DeviceType,
             typename IndexType >
bool navierStokesSetter< MeshType, SolverStarter > :: run( const tnlParameterContainer& parameters )
{
   std::cerr << "The solver is not implemented for the mesh " << MeshType::getType() << "." << std::endl;
   return false;
}

template< typename MeshReal, typename Device, typename MeshIndex, typename SolverStarter >
template< typename RealType,
          typename DeviceType,
          typename IndexType >
bool navierStokesSetter< tnlGrid< 2, MeshReal, Device, MeshIndex >, SolverStarter >::run( const tnlParameterContainer& parameters )
{
   SolverStarter solverStarter;
   const tnlString& schemeName = parameters. getParameter< tnlString >( "scheme" );
   if( schemeName == "lax-fridrichs" )
      return solverStarter. run< navierStokesSolver< MeshType,
                                                     tnlLaxFridrichs< MeshType,
                                                                     tnlCentralFDMGradient< MeshType > > > >
                                                     ( parameters );
};

#endif /* NAVIERSTOKESSETTER_IMPL_H_ */
