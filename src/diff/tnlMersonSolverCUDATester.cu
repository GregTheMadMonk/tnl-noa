/***************************************************************************
                          tnlMersonSolverCUDATester.cu
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
 
 #include <diff/tnlMersonSolverCUDATester.cu.h>
 
 void heatEquationRHS( const int gridDimX,
                       const int gridDimY,
                       const int blockDimX,
                       const int blockDimY,
                       const int xSize,
                       const int ySize,
                       const float& hX,
                       const float& hY,
                       const float* u,
                       float* fu )
{
   dim3 gridDim( gridDimX, gridDimY );
   dim3 blockDim( blockDimX, blockDimY );
   heatEquationRHSKernel<<< gridDim, blockDim >>>( xSize, ySize, hX, hY, u, fu );
}

void heatEquationRHS( const int gridDimX,
                      const int gridDimY,
                      const int blockDimX,
                      const int blockDimY,
                      const int xSize,
                      const int ySize,
                      const double& hX,
                      const double& hY,
                      const double* u,
                      double* fu )
{
   dim3 gridDim( gridDimX, gridDimY );
   dim3 blockDim( blockDimX, blockDimY );
   heatEquationRHSKernel<<< gridDim, blockDim >>>( xSize, ySize, hX, hY, u, fu );
}