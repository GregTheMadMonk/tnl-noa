/***************************************************************************
                          MIC.h  -  description
                          -------------------
    begin                : Nov 7, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Vit Hanousek

#pragma once

#include <iostream>
#include <cstring>
#include <unistd.h>
#include <TNL/String.h>
#include <TNL/Assert.h>
#include <TNL/Pointers/SmartPointersRegister.h>
#include <TNL/Timer.h>

#include <TNL/Devices/CudaCallable.h>


namespace TNL {
namespace Devices {
namespace {

//useful macros from Intel's tutorials -- but we do not use it, becaouse it is tricky (system of maping variables CPU-MIC)
#define ALLOC alloc_if(1) //alloac variable at begining of offloaded block -- default
#define FREE free_if(1) // delete variable at the end of offloaded block -- default
#define RETAIN free_if(0) //do not delete variable at the end of offladed block
#define REUSE alloc_if(0) //do not alloc variable at begin of offloaded block, reuse variable on MIC which was not deleted befeore

//structure which hides pointer - bypass mapping of variables and addresses of arrays and allow get RAW addres of MIC memory to RAM
template< typename Type >
struct MICHider{
    Type *pointer;
};

//inflatable structure -- structures can be copied to MIC - classes not (viz paper published after CSJP 2016 in Krakow)
//object can be copied in side this structure and then copied into MIC memory
template <unsigned int VELIKOST>
struct MICStruct{
	uint8_t data[VELIKOST];
};

//Macros which can make code better readeble --but they are tricky, creating variables with specific names...
//version using inflatable structure
#define TNLMICSTRUCT(bb,typ) Devices::MICStruct<sizeof(typ)> s ## bb; \
                             memcpy((void*)& s ## bb,(void*)& bb,sizeof(typ));
#define TNLMICSTRUCTOFF(bb,typ) s ## bb
#define TNLMICSTRUCTUSE(bb,typ) typ * kernel ## bb = (typ*) &s ## bb;
#define TNLMICSTRUCTALLOC(bb,typ) typ * kernel ## bb = (typ*) malloc (sizeof(typ)); \
                                memcpy((void*)kernel ## bb,(void*) & s ## bb, sizeof(typ));

//version which retypes pointer of object to pointer to array of uint8_t,
//object can be copied using uint8_t pointer as array with same length as object size
#define TNLMICHIDE(bb,typ) uint8_t * u ## bb=(uint8_t *)&bb; \
                           MICHider<typ> kernel ## bb;
#define TNLMICHIDEALLOCOFF(bb,typ) in(u ## bb:length(sizeof(typ))) out(kernel ## bb)
#define TNLMICHIDEALLOC(bb,typ) kernel ## bb.pointer=(typ*)malloc(sizeof(typ)); \
                                memcpy((void*)kernel ## bb.pointer,(void*)u ## bb,sizeof(typ));
#define TNLMICHIDEFREEOFF(bb,typ) in(kernel ## bb)
#define TNLMICHIDEFREE(bb,typ) free((void*)kernel ## bb.pointer

class MIC
{
   public:

        static String getDeviceType()
        {
            return String( "Devices::MIC" );
        };

#ifdef HAVE_MIC

       //useful debuging -- but produce warning
       __cuda_callable__ static inline void CheckMIC(void)
       {
            #ifdef __MIC__
                    std::cout<<"ON MIC"<<std::endl;
            #else
                    std::cout<<"ON CPU" <<std::endl;
            #endif
        };


        //old copying funciton  -- deprecated
        template <typename TYP>
        static
        TYP * passToDevice(TYP &objektCPU)
        {
                uint8_t * uk=(uint8_t *)&objektCPU;
                MICHider<TYP> ret;

                #pragma offload target(mic) in(uk:length(sizeof(TYP))) out(ret)
                {
                    ret.pointer=(TYP*)malloc(sizeof(TYP));
                    std::memcpy((void*)ret.pointer,(void*)uk,sizeof(TYP));
                }
                return ret.pointer;

                std::cout << "Někdo mně volá :-D" <<std::endl;
        };

        //old cleaning function -- deprecated
        template <typename TYP>
        static
        void freeFromDevice(TYP *objektMIC)
        {
            MICHider<TYP> ptr;
            ptr.pointer=objektMIC;
            #pragma offload target(mic) in(ptr)
            {
                free((void*)ptr.pointer);
            }
        };

        static inline
        void CopyToMIC(void* mic_ptr,void* ptr,size_t size)
        {
            uint8_t image[size];
            std::memcpy((void*)&image,ptr,size);
            Devices::MICHider<void> hide_ptr;
            hide_ptr.pointer=mic_ptr;
            #pragma offload target(mic) in(hide_ptr) in(image) in(size)
            {
                std::memcpy((void*)hide_ptr.pointer,(void*)&image,size);
            }
        };

        static inline
        void* AllocMIC(size_t size)
        {
            Devices::MICHider<void> hide_ptr;
            #pragma offload target(mic) out(hide_ptr) in(size)
            {
                hide_ptr.pointer=malloc(size);
            }
            return hide_ptr.pointer;
        };

        static inline
        void FreeMIC(void* ptr)
        {
                Devices::MICHider<void> hide_ptr;
                hide_ptr.pointer=ptr;
                #pragma offload target(mic) in(hide_ptr)
                {
                        free(hide_ptr.pointer);
                }
        };


#endif

   static void insertSmartPointer( Pointers::SmartPointer* pointer )
   {
      smartPointersRegister.insert( pointer, -1 );
   }

   static void removeSmartPointer( Pointers::SmartPointer* pointer )
   {
      smartPointersRegister.remove( pointer, -1 );
   }

   // Negative deviceId means that CudaDeviceInfo::getActiveDevice will be
   // called to get the device ID.
   static bool synchronizeDevice( int deviceId = -1 )
   {
      smartPointersSynchronizationTimer.start();
      bool b = smartPointersRegister.synchronizeDevice( deviceId );
      smartPointersSynchronizationTimer.stop();
      return b;
   }

   static Timer smartPointersSynchronizationTimer;

protected:
   static Pointers::SmartPointersRegister smartPointersRegister;
};

Pointers::SmartPointersRegister MIC::smartPointersRegister;
Timer MIC::smartPointersSynchronizationTimer;

} // namespace <unnamed>
} // namespace Devices
} // namespace TNL
