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
#include <string.h>


#ifndef TNLMIC_H
#define TNLMIC_H

#define ALLOC alloc_if(1) //naalokuj promenou na zacatku offload  bloku -- default
#define FREE free_if(1) // smaz promenou na konci offload bloku -- default
#define RETAIN free_if(0) //nesmaz promenou na konci bloku
#define REUSE alloc_if(0) //nealokuj proměnnou na zacatku

//useful if you have an adress to MIC memory
template< typename Type >
struct satanHider{
    Type *pointer;
};


class tnlMIC
{
   public:
       
        enum { DeviceType = tnlMICDevice };
        
        
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
         
         //useful function to pass object in mělký copy onto MIC. To adress the object use the base adress of object
        //this function use MIC adress translation tables.
        
        /*template <typename TYP>
        static
        TYP * passToDevice(TYP &objektCPU)
        {
                uint8_t * uk=(uint8_t *)&objektCPU; 
                satanHider<TYP> ret;
                
                #pragma offload target(mic) in(uk:length(sizeof(TYP)) ALLOC RETAIN) out(ret)
                {
                    ret.pointer=(TYP*)&uk;
                }
                return ret.pointer;
        }
         
        template <typename TYP>
        static
        void freeFromDevice(TYP *objektCPU)
        {
                #pragma offload target(mic) in(objektCPU:length(0) REUSE FREE)
                {
                }
        }

         */
         
        template <typename TYP>
        static
        TYP * passToDevice(TYP &objektCPU)
        {
                uint8_t * uk=(uint8_t *)&objektCPU; 
                satanHider<TYP> ret;
                
                #pragma offload target(mic) in(uk:length(sizeof(TYP))) out(ret)
                {
                    ret.pointer=(TYP*)malloc(sizeof(TYP));
                    memcpy((void*)ret.pointer,(void*)uk,sizeof(TYP));
                }
                return ret.pointer;
        }
        
        template <typename TYP>
        static
        void freeFromDevice(TYP *objektMIC)
        {
            satanHider<TYP> ptr;
            ptr.pointer=objektMIC;
            #pragma offload target(mic) in(ptr)
            {
                free((void*)ptr.pointer);
            }
        }

};







#endif /* TNLMIC_H */

