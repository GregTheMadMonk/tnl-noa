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

template< typename RealType,
          typename DeviceType,
          typename IndexType,
          typename MeshType,
          typename ConfigTag,
          typename SolverStarter >
bool simpleProblemSetter< RealType, DeviceType, IndexType, MeshType, ConfigTag, SolverStarter > :: run( const tnlParameterContainer& parameters )
{
   SolverStarter solverStarter;
   return solverStarter. template run< simpleProblemSolver< MeshType > >( parameters );
}


#endif /* SIMPLEPROBLEMSETTER_IMPL_H_ */
