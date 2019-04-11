/***************************************************************************
                          CudaCallable.h  -  description
                             -------------------
    begin                : Jun 20, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

// The __cuda_callable__ macro has to be in a separate header file to avoid
// infinite loops by the #include directives.
//
// For example, the implementation of Devices::Cuda needs TNL_ASSERT_*
// macros, which need __cuda_callable__ functions.

/***
 * This macro serves for definition of function which are supposed to be called
 * even from device. If HAVE_CUDA is defined, the __cuda_callable__ function
 * is compiled for both CPU and GPU. If HAVE_CUDA is not defined, this macro has
 * no effect. Support for Intel Xeon Phi is now in "hibernated" state.
 */
#ifdef HAVE_MIC 
   #define __cuda_callable__ __attribute__((target(mic)))
#elif HAVE_CUDA
   #define __cuda_callable__ __device__ __host__
#else
   #define __cuda_callable__
#endif
