/***************************************************************************
                          tnl-cuda-kernels.cu.h  -  description
                             -------------------
    begin                : Jan 19, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
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

#ifndef TNLCUDAKERNELS_CU_H_
#define TNLCUDAKERNELS_CU_H_

int tnlCUDAReductionMin( const int size,
                         const int* input,
                         int& result,
                         int* device_aux_array = 0 );

int tnlCUDAReductionMax( const int size,
                         const int* input,
                         int& result,
                         int* device_aux_array = 0 );

int tnlCUDAReductionSum( const int size,
                         const int* input,
                         int& result,
                         int* device_aux_array = 0 );

bool tnlCUDAReductionMin( const int size,
                          const float* input,
                          float& result,
                          float* device_aux_array = 0 );

bool tnlCUDAReductionMax( const int size,
                          const float* input,
                          float& result,
                          float* device_aux_array = 0 );

bool tnlCUDAReductionSum( const int size,
                          const float* input,
                          float& result,
                          float* device_aux_array = 0 );

bool tnlCUDAReductionMin( const int size,
                          const double* input,
                          double& result,
                          double* device_aux_array = 0 );

bool tnlCUDAReductionMax( const int size,
                          const double* input,
                          double& result,
                          double* device_aux_array = 0 );

bool tnlCUDAReductionSum( const int size,
                          const double* input,
                          double& result,
                          double* device_aux_array = 0 );

/*
 * Simple reduction 5
 */
bool tnlCUDASimpleReduction5Min( const int size,
                                 const int* input,
                                 int& result );
bool tnlCUDASimpleReduction5Max( const int size,
                                 const int* input,
                                 int& result );
bool tnlCUDASimpleReduction5Sum( const int size,
                                 const int* input,
                                 int& result );
bool tnlCUDASimpleReduction5Min( const int size,
                                 const float* input,
                                 float& result);
bool tnlCUDASimpleReduction5Max( const int size,
                                 const float* input,
                                 float& result);
bool tnlCUDASimpleReduction5Sum( const int size,
                                 const float* input,
                                 float& result);
bool tnlCUDASimpleReduction5Min( const int size,
                                 const double* input,
                                 double& result);
bool tnlCUDASimpleReduction5Max( const int size,
                                 const double* input,
                                 double& result );
bool tnlCUDASimpleReduction5Sum( const int size,
                                 const double* input,
                                 double& result );

/*
 * Simple reduction 4
 */
bool tnlCUDASimpleReduction4Min( const int size,
                                 const int* input,
                                 int& result );
bool tnlCUDASimpleReduction4Max( const int size,
                                 const int* input,
                                 int& result );
bool tnlCUDASimpleReduction4Sum( const int size,
                                 const int* input,
                                 int& result );
bool tnlCUDASimpleReduction4Min( const int size,
                                 const float* input,
                                 float& result);
bool tnlCUDASimpleReduction4Max( const int size,
                                 const float* input,
                                 float& result);
bool tnlCUDASimpleReduction4Sum( const int size,
                                 const float* input,
                                 float& result);
bool tnlCUDASimpleReduction4Min( const int size,
                                 const double* input,
                                 double& result);
bool tnlCUDASimpleReduction4Max( const int size,
                                 const double* input,
                                 double& result );
bool tnlCUDASimpleReduction4Sum( const int size,
                                 const double* input,
                                 double& result );

/*
 * Simple reduction 3
 */
bool tnlCUDASimpleReduction3Min( const int size,
                          const int* input,
                          int& result );
bool tnlCUDASimpleReduction3Max( const int size,
                          const int* input,
                          int& result );
bool tnlCUDASimpleReduction3Sum( const int size,
                          const int* input,
                          int& result );
bool tnlCUDASimpleReduction3Min( const int size,
                            const float* input,
                            float& result);
bool tnlCUDASimpleReduction3Max( const int size,
                            const float* input,
                            float& result);
bool tnlCUDASimpleReduction3Sum( const int size,
                            const float* input,
                            float& result);
bool tnlCUDASimpleReduction3Min( const int size,
                             const double* input,
                             double& result);
bool tnlCUDASimpleReduction3Max( const int size,
                             const double* input,
                             double& result );
bool tnlCUDASimpleReduction3Sum( const int size,
                             const double* input,
                             double& result );

/*
 * Simple reduction 2
 */
bool tnlCUDASimpleReduction2Min( const int size,
                          const int* input,
                          int& result );
bool tnlCUDASimpleReduction2Max( const int size,
                          const int* input,
                          int& result );
bool tnlCUDASimpleReduction2Sum( const int size,
                          const int* input,
                          int& result );
bool tnlCUDASimpleReduction2Min( const int size,
                            const float* input,
                            float& result);
bool tnlCUDASimpleReduction2Max( const int size,
                            const float* input,
                            float& result);
bool tnlCUDASimpleReduction2Sum( const int size,
                            const float* input,
                            float& result);
bool tnlCUDASimpleReduction2Min( const int size,
                             const double* input,
                             double& result);
bool tnlCUDASimpleReduction2Max( const int size,
                             const double* input,
                             double& result );
bool tnlCUDASimpleReduction2Sum( const int size,
                             const double* input,
                             double& result );

/*
 * Simple reduction 1
 */
bool tnlCUDASimpleReduction1Min( const int size,
                          const int* input,
                          int& result );
bool tnlCUDASimpleReduction1Max( const int size,
                          const int* input,
                          int& result );
bool tnlCUDASimpleReduction1Sum( const int size,
                          const int* input,
                          int& result );
bool tnlCUDASimpleReduction1Min( const int size,
                            const float* input,
                            float& result);
bool tnlCUDASimpleReduction1Max( const int size,
                            const float* input,
                            float& result);
bool tnlCUDASimpleReduction1Sum( const int size,
                            const float* input,
                            float& result);
bool tnlCUDASimpleReduction1Min( const int size,
                             const double* input,
                             double& result);
bool tnlCUDASimpleReduction1Max( const int size,
                             const double* input,
                             double& result );
bool tnlCUDASimpleReduction1Sum( const int size,
                             const double* input,
                             double& result );


#endif /* TNLCUDAKERNELS_CU_H_ */
