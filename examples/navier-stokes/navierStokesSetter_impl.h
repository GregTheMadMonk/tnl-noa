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

#include <mesh/tnlGrid.h>
#include <schemes/euler/fvm/tnlLaxFridrichs.h>
#include <schemes/gradient/tnlCentralFDMGradient.h>

template< typename SolverStarter >
   template< typename RealType,
             typename DeviceType,
             typename IndexType >
bool navierStokesSetter< SolverStarter > :: run( const tnlParameterContainer& parameters ) const
{
   int dimensions = parameters. GetParameter< int >( "dimensions" );
   if( dimensions != 2 )
   {
      cerr << "The problem is not defined for " << dimensions << "dimensions." << endl;
      return false;
   }
   SolverStarter solverStarter;
   const tnlString& schemeName = parameters. GetParameter< tnlString >( "scheme" );
   if( dimensions == 2 )
   {
      typedef tnlGrid< 2, RealType, DeviceType, IndexType > MeshType;
      if( schemeName == "lax-fridrichs" )
         return solverStarter. run< navierStokesSolver< MeshType,
                                                        tnlLaxFridrichs< MeshType,
                                                                        tnlCentralFDMGradient< MeshType > > > >
                                                        ( parameters );
   }
}


#endif /* NAVIERSTOKESSETTER_IMPL_H_ */
