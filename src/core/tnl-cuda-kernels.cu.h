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
                                 int& result,
                                 int* device_aux = 0 );
bool tnlCUDASimpleReduction5Max( const int size,
                                 const int* input,
                                 int& result,
                                 int* device_aux = 0 );
bool tnlCUDASimpleReduction5Sum( const int size,
                                 const int* input,
                                 int& result,
                                 int* device_aux = 0 );
bool tnlCUDASimpleReduction5Min( const int size,
                                 const float* input,
                                 float& result,
                                 float* device_aux = 0 );
bool tnlCUDASimpleReduction5Max( const int size,
                                 const float* input,
                                 float& result,
                                 float* device_aux = 0 );
bool tnlCUDASimpleReduction5Sum( const int size,
                                 const float* input,
                                 float& result,
                                 float* device_aux = 0 );
bool tnlCUDASimpleReduction5Min( const int size,
                                 const double* input,
                                 double& result,
                                 double* device_aux = 0 );
bool tnlCUDASimpleReduction5Max( const int size,
                                 const double* input,
                                 double& result,
                                 double* device_aux = 0  );
bool tnlCUDASimpleReduction5Sum( const int size,
                                 const double* input,
                                 double& result,
                                 double* device_aux = 0  );

/*
 * Simple reduction 4
 */
bool tnlCUDASimpleReduction4Min( const int size,
                                 const int* input,
                                 int& result,
                                 int* device_aux = 0  );
bool tnlCUDASimpleReduction4Max( const int size,
                                 const int* input,
                                 int& result,
                                 int* device_aux = 0  );
bool tnlCUDASimpleReduction4Sum( const int size,
                                 const int* input,
                                 int& result,
                                 int* device_aux = 0  );
bool tnlCUDASimpleReduction4Min( const int size,
                                 const float* input,
                                 float& result,
                                 float* device_aux = 0 );
bool tnlCUDASimpleReduction4Max( const int size,
                                 const float* input,
                                 float& result,
                                 float* device_aux = 0 );
bool tnlCUDASimpleReduction4Sum( const int size,
                                 const float* input,
                                 float& result,
                                 float* device_aux = 0 );
bool tnlCUDASimpleReduction4Min( const int size,
                                 const double* input,
                                 double& result,
                                 double* device_aux = 0 );
bool tnlCUDASimpleReduction4Max( const int size,
                                 const double* input,
                                 double& result,
                                 double* device_aux = 0  );
bool tnlCUDASimpleReduction4Sum( const int size,
                                 const double* input,
                                 double& result,
                                 double* device_aux = 0  );

/*
 * Simple reduction 3
 */
bool tnlCUDASimpleReduction3Min( const int size,
                                 const int* input,
                                 int& result,
                                 int* device_aux = 0  );
bool tnlCUDASimpleReduction3Max( const int size,
                                 const int* input,
                                 int& result,
                                 int* device_aux = 0  );
bool tnlCUDASimpleReduction3Sum( const int size,
                                 const int* input,
                                 int& result,
                                 int* device_aux = 0  );
bool tnlCUDASimpleReduction3Min( const int size,
                                 const float* input,
                                 float& result,
                                 float* device_aux = 0 );
bool tnlCUDASimpleReduction3Max( const int size,
                                 const float* input,
                                 float& result,
                                 float* device_aux = 0 );
bool tnlCUDASimpleReduction3Sum( const int size,
                                 const float* input,
                                 float& result,
                                 float* device_aux = 0 );
bool tnlCUDASimpleReduction3Min( const int size,
                                 const double* input,
                                 double& result,
                                 double* device_aux = 0 );
bool tnlCUDASimpleReduction3Max( const int size,
                                 const double* input,
                                 double& result,
                                 double* device_aux = 0  );
bool tnlCUDASimpleReduction3Sum( const int size,
                                 const double* input,
                                 double& result,
                                 double* device_aux = 0  );

/*
 * Simple reduction 2
 */
bool tnlCUDASimpleReduction2Min( const int size,
                                 const int* input,
                                 int& result,
                                 int* device_aux = 0  );
bool tnlCUDASimpleReduction2Max( const int size,
                                 const int* input,
                                 int& result,
                                 int* device_aux = 0  );
bool tnlCUDASimpleReduction2Sum( const int size,
                                 const int* input,
                                 int& result,
                                 int* device_aux = 0  );
bool tnlCUDASimpleReduction2Min( const int size,
                                 const float* input,
                                 float& result,
                                 float* device_aux = 0 );
bool tnlCUDASimpleReduction2Max( const int size,
                                 const float* input,
                                 float& result,
                                 float* device_aux = 0 );
bool tnlCUDASimpleReduction2Sum( const int size,
                                 const float* input,
                                 float& result,
                                 float* device_aux = 0 );
bool tnlCUDASimpleReduction2Min( const int size,
                                 const double* input,
                                 double& result,
                                 double* device_aux = 0 );
bool tnlCUDASimpleReduction2Max( const int size,
                                 const double* input,
                                 double& result,
                                 double* device_aux = 0  );
bool tnlCUDASimpleReduction2Sum( const int size,
                                 const double* input,
                                 double& result,
                                 double* device_aux = 0  );

/*
 * Simple reduction 1
 */
bool tnlCUDASimpleReduction1Min( const int size,
                                 const int* input,
                                 int& result,
                                 int* device_aux = 0 );
bool tnlCUDASimpleReduction1Max( const int size,
                                 const int* input,
                                 int& result,
                                 int* device_aux = 0 );
bool tnlCUDASimpleReduction1Sum( const int size,
                                 const int* input,
                                 int& result,
                                 int* device_aux = 0 );
bool tnlCUDASimpleReduction1Min( const int size,
                                 const float* input,
                                 float& result,
                                 float* device_aux = 0 );
bool tnlCUDASimpleReduction1Max( const int size,
                                 const float* input,
                                 float& result,
                                 float* device_aux = 0 );
bool tnlCUDASimpleReduction1Sum( const int size,
                                 const float* input,
                                 float& result,
                                 float* device_aux = 0 );
bool tnlCUDASimpleReduction1Min( const int size,
                                 const double* input,
                                 double& result,
                                 double* device_aux = 0 );
bool tnlCUDASimpleReduction1Max( const int size,
                                 const double* input,
                                 double& result,
                                 double* device_aux = 0 );
bool tnlCUDASimpleReduction1Sum( const int size,
                                 const double* input,
                                 double& result,
                                 double* device_aux = 0 );


/********************************************************************/

int testReduction1( int size,
                    int* drp_input,
                    int* drp_output,
                    int* output );

int testReduction2( int size,
                    int* drp_input,
                    int* drp_output,
                    int* output );

int testReduction3( int size,
                    int* drp_input,
                    int* drp_output,
                    int* output );

int testReduction4( int size,
                    int* drp_input,
                    int* drp_output,
                    int* output );

void reductionKernel5Switch(int* dp_output, int* dp_input, uint gridSz, uint blockSz, uint shmemBB);


int testReduction5( int size,
                    int* drp_input,
                    int* drp_output,
                    int* output );

inline void reductionKernel6Switch(uint psize, int* dp_output, int* dp_input, uint gridSz, uint blockSz, uint shmemBB);

int testReduction6( int size,
                    int* drp_input,
                    int* drp_output,
                    int* output );



#endif /* TNLCUDAKERNELS_CU_H_ */
