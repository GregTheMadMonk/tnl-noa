/***************************************************************************
                          simpleProblemSetter_impl.h  -  description
                             -------------------
    begin                : Mar 10, 2013
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

#ifndef SIMPLEPROBLEMSETTER_IMPL_H_
#define SIMPLEPROBLEMSETTER_IMPL_H_

template< typename SolverStarter >
   template< typename RealType,
             typename DeviceType,
             typename IndexType >
bool simpleProblemSetter< SolverStarter > :: run( const tnlParameterContainer& parameters ) const
{
   int dimensions = parameters. GetParameter< int >( "dimensions" );
   if( dimensions <= 0 || dimensions > 3 )
   {
      cerr << "The problem is not defined for " << dimensions << "dimensions." << endl;
      return false;
   }
   SolverStarter solverStarter;
   if( dimensions == 1 )
   {
      typedef tnlGrid< 1, RealType, DeviceType, IndexType > MeshType;
      return solverStarter. run< simpleProblemSolver< MeshType > >( parameters );
   }
   if( dimensions == 2 )
   {
      typedef tnlGrid< 2, RealType, DeviceType, IndexType > MeshType;
      return solverStarter. run< simpleProblemSolver< MeshType > >( parameters );
   }
   if( dimensions == 3 )
   {
      typedef tnlGrid< 3, RealType, DeviceType, IndexType > MeshType;
      return solverStarter. run< simpleProblemSolver< MeshType > >( parameters );
   }
}


#endif /* SIMPLEPROBLEMSETTER_IMPL_H_ */
