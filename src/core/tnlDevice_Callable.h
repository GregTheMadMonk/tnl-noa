/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   tnldevice_callable.h
 * Author: hanouvit
 *
 * Created on 18. dubna 2016, 15:49
 */

#ifndef TNLDEVICE_CALLABLE_H
#define TNLDEVICE_CALLABLE_H
/*
//deprecated __cuda_callable__
#ifdef HAVE_CUDA
    #define __cuda_callable__ __device__ __host__
#else
    #define __cuda_callable__ 
#endif

//NEW, better  __device_callable__ --used only with MIC touch code
#ifdef HAVE_MIC 
    #define __device_callable__ __attribute__((target(mic)))
#elif HAVE_CUDA
    #define __device_callable__ __device__ __host__
#else
    #define __device_callable__ 
#endif
*/

#define __cuda_callable__ __attribute__((target(mic)))
#define __device_callable__ __attribute__((target(mic)))
#endif /* TNLDEVICE_CALLABLE_H */

