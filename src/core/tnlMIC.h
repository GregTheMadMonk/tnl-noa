/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   tnlMIC.h
 * Author: hanouvit
 *
 * Created on 18. dubna 2016, 12:38
 */

#include <iostream>
#include <core/tnlString.h>
#include <core/tnlDevice.h>
#include <core/tnlDevice_Callable.h>


#ifndef TNLMIC_H
#define TNLMIC_H

#define ALLOC alloc_if(1) //naalokuj promenou na zacatku offload  bloku -- default
#define FREE free_if(1) // smaz promenou na konci offload bloku -- default
#define RETAIN free_if(0) //nesmaz promenou na konci bloku
#define REUSE alloc_if(0) //nealokuj proměnnou na zacatku


class tnlMIC
{
   public:
       //useful debuging -- but produce warning
       __device_callable__ static inline void CheckMIC(void)
       {
            #ifdef __MIC__
                    std::cout<<"ON MIC"<<std::endl;
            #else
                    std::cout<<"ON CPU" <<std::endl;
            #endif
        }
       
       
        static tnlString getDeviceType()
        {
            return tnlString( "tnlMIC" );
        }

         __device_callable__ static inline tnlDeviceEnum getDevice()
         {
             return tnlMICDevice;
         }
};

template< typename Type >
struct satanHider{
    Type *pointer;
};

#endif /* TNLMIC_H */

