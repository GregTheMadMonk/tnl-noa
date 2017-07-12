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

#ifdef HAVE_CUDA
#define __cuda_callable__ __device__ __host__
#else
#define __cuda_callable__
#endif
