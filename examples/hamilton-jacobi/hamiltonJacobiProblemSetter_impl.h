/***************************************************************************
                          hamiltonJacobiProblemSetter_impl.h  -  description
                             -------------------
    begin                : Jul 8 , 2014
    copyright            : (C) 2014 by Tomas Sobotik
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef HAMILTONJACOBIPROBLEMSETTER_IMPL_H_
#define HAMILTONJACOBIPROBLEMSETTER_IMPL_H_

template< typename RealType,
          typename DeviceType,
          typename IndexType,
          typename MeshType,
          typename ConfigTag,
          typename SolverStarter >
bool hamiltonJacobiProblemSetter< RealType, DeviceType, IndexType, MeshType, ConfigTag, SolverStarter > :: run( const tnlParameterContainer& parameters )
{
	enum { Dimensions = MeshType::Dimensions };



		if( Dimensions <= 0 || Dimensions > 3 )
		{
		   cerr << "The problem is not defined for " << Dimensions << "dimensions." << endl;
		   return false;
		}
		else
		{
		      typedef tnlConstantFunction< Dimensions, RealType > RightHandSide;
		      typedef tnlStaticVector < Dimensions, RealType > Vertex;
		      typedef tnlNeumannReflectionBoundaryConditions< MeshType,  RealType, IndexType > BoundaryConditions;

		      SolverStarter solverStarter;

			const tnlString& schemeName = parameters. GetParameter< tnlString >( "scheme" );

			if( schemeName == "upwind" )
			{
		        typedef upwindEikonalScheme< MeshType, RealType, IndexType  > Operator;
		        typedef hamiltonJacobiProblemSolver< MeshType, Operator, BoundaryConditions, RightHandSide > Solver;
		        return solverStarter.template run< Solver >( parameters );
			}
			else if ( schemeName == "godunov")
			{
				typedef godunovEikonalScheme< MeshType, RealType, IndexType  > Operator;
				typedef hamiltonJacobiProblemSolver< MeshType, Operator, BoundaryConditions, RightHandSide > Solver;
				return solverStarter.template run< Solver >( parameters );
			}
			else
			   cerr << "Unknown scheme '" << schemeName << "'." << endl;


			return false;
		}
}


#endif /* HAMILTONJACOBIPROBLEMSETTER_IMPL_H_ */
