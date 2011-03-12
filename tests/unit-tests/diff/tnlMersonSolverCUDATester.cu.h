/***************************************************************************
                          tnlMersonSolverCUDATester.cu.h
                             -------------------
    begin                : Feb 2, 2010
    copyright            : (C) 2009 by Tomas Oberhuber
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



#ifndef TNLMERSONSOLVERCUDATESTER_CU_H_
#define TNLMERSONSOLVERCUDATESTER_CU_H_

template< class T > __global__ void heatEquationRHSKernel( const int xSize,
		                                                   const int ySize,
		                                                   const T hX,
		                                                   const T hY,
		                                                   const T* u,
		                                                   T* fu )
{
	const int i = blockIdx. x * blockDim. x + threadIdx. x;
	const int j = blockIdx. y * blockDim. y + threadIdx. y;

	if( i == 0 || j == 0 || i == xSize - 1 || j == ySize - 1 )
		fu[ xSize * j + i ] = 0.0;
	else
		if( i < xSize && j < ySize )
		fu[ xSize * j + i ] = ( u[ xSize * j + i + 1 ] - 2.0 * u[ xSize * j + i ] + u[ xSize * j + i - 1] ) / ( hX * hX ) +
	                          ( u[ xSize * j + i + xSize ] - 2.0 * u[ xSize * j + i ] + u[ xSize * j + i - xSize ] ) / ( hY * hY );

}


#endif /* TNLMERSONSOLVERCUDATESTER_CU_H_ */
