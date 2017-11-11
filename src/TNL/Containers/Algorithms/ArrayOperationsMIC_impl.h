/***************************************************************************
                          ArrayOperationsMIC_impl.h  -  description
                             -------------------
    begin                : Mar 4, 2017
    copyright            : (C) 2017 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Vit Hanousek

#pragma once

#include <iostream>

#include <TNL/tnlConfig.h>
#include <TNL/Math.h>
#include <TNL/Exceptions/MICSupportMissing.h>
#include <TNL/Exceptions/MICBadAlloc.h>
#include <TNL/Containers/Algorithms/ArrayOperations.h>
#include <TNL/Containers/Algorithms/Reduction.h>
#include <TNL/Containers/Algorithms/ReductionOperations.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

static constexpr std::size_t MIC_STACK_VAR_LIM = 5*1024*1024;

template< typename Element, typename Index >
void
ArrayOperations< Devices::MIC >::
allocateMemory( Element*& data,
                const Index size )
{
#ifdef HAVE_MIC
   data = (Element*) Devices::MIC::AllocMIC( size * sizeof(Element) );
   if( ! data )
      throw Exceptions::MICBadAlloc();
#else
   throw Exceptions::MICSupportMissing();
#endif
}

template< typename Element >
void
ArrayOperations< Devices::MIC >::
freeMemory( Element* data )
{
   TNL_ASSERT( data, );
#ifdef HAVE_MIC
   Devices::MIC::FreeMIC( data );
#else
   throw Exceptions::MICSupportMissing();
#endif
}

template< typename Element >
void
ArrayOperations< Devices::MIC >::
setMemoryElement( Element* data,
                  const Element& value )
{
   TNL_ASSERT( data, );
   ArrayOperations< Devices::MIC >::setMemory( data, value, 1 );
}

template< typename Element >
Element
ArrayOperations< Devices::MIC >::
getMemoryElement( const Element* data )
{
   TNL_ASSERT( data, );
   Element result;
   ArrayOperations< Devices::Host, Devices::MIC >::copyMemory< Element, Element, int >( &result, data, 1 );
   return result;
}

template< typename Element, typename Index >
Element&
ArrayOperations< Devices::MIC >::
getArrayElementReference( Element* data, const Index i )
{
   TNL_ASSERT( data, );
   return data[ i ];
}

template< typename Element, typename Index >
const
Element& ArrayOperations< Devices::MIC >::
getArrayElementReference( const Element* data, const Index i )
{
   TNL_ASSERT( data, );
   return data[ i ];
}

template< typename Element, typename Index >
bool
ArrayOperations< Devices::MIC >::
setMemory( Element* data,
           const Element& value,
           const Index size )
{
   TNL_ASSERT( data, );
#ifdef HAVE_MIC
   Element tmp=value;
   Devices::MICHider<Element> hide_ptr;
   hide_ptr.pointer=data;
   #pragma offload target(mic) in(hide_ptr,tmp,size)
   {
       Element * dst= hide_ptr.pointer;
       for(int i=0;i<size;i++)
           dst[i]=tmp;
   }
   return true;
#else
   throw Exceptions::MICSupportMissing();
#endif
}

template< typename DestinationElement,
          typename SourceElement,
          typename Index >
bool
ArrayOperations< Devices::MIC >::
copyMemory( DestinationElement* destination,
            const SourceElement* source,
            const Index size )
{
   TNL_ASSERT( destination, );
   TNL_ASSERT( source, );
   #ifdef HAVE_MIC
      if( std::is_same< DestinationElement, SourceElement >::value )
      {
         Devices::MICHider<void> src_ptr;
         src_ptr.pointer=(void*)source;
         Devices::MICHider<void> dst_ptr;
         dst_ptr.pointer=(void*)destination;
         #pragma offload target(mic) in(src_ptr,dst_ptr,size)
         {
             memcpy(dst_ptr.pointer,src_ptr.pointer,size*sizeof(DestinationElement));
         }
         return true;
      }
      else
      {
         Devices::MICHider<const SourceElement> src_ptr;
         src_ptr.pointer=source;
         Devices::MICHider<DestinationElement> dst_ptr;
         dst_ptr.pointer=destination;
         #pragma offload target(mic) in(src_ptr,dst_ptr,size)
         {
             for(int i=0;i<size;i++)
                 dst_ptr.pointer[i]=src_ptr.pointer[i];
         }
         return true;

      }
   #else
      throw Exceptions::MICSupportMissing();
   #endif
      return false;
}

template< typename Element1,
          typename Element2,
          typename Index >
bool
ArrayOperations< Devices::MIC >::
compareMemory( const Element1* destination,
               const Element2* source,
               const Index size )
{
   TNL_ASSERT( destination, );
   TNL_ASSERT( source, );
#ifdef HAVE_MIC
   if( std::is_same< Element1, Element2 >::value )
   {
      Devices::MICHider<void> src_ptr;
      src_ptr.pointer=(void*)source;
      Devices::MICHider<void> dst_ptr;
      dst_ptr.pointer=(void*)destination;
      int ret=0;
      #pragma offload target(mic) in(src_ptr,dst_ptr,size) out(ret)
      {
          ret=memcmp(dst_ptr.pointer,src_ptr.pointer,size*sizeof(Element1));
      }
      if(ret==0)
          return true;
   }
   else
   {
      Devices::MICHider<const Element2> src_ptr;
      src_ptr.pointer=source;
      Devices::MICHider<const Element1> dst_ptr;
      dst_ptr.pointer=destination;
      bool ret=false;
      #pragma offload target(mic) in(src_ptr,dst_ptr,size) out(ret)
      {
          int i=0;
          for(i=0;i<size;i++)
              if(dst_ptr.pointer[i]!=src_ptr.pointer[i])
                  break;
          if(i==size)
              ret=true;
          else
              ret=false;
      }
      return ret;
   }
   return false;
#else
   throw Exceptions::MICSupportMissing();
#endif
}

/****
 * Operations MIC -> Host
 */

template< typename DestinationElement,
          typename SourceElement,
          typename Index >
bool
ArrayOperations< Devices::Host, Devices::MIC >::
copyMemory( DestinationElement* destination,
            const SourceElement* source,
            const Index size )
{
   TNL_ASSERT( destination, );
   TNL_ASSERT( source, );
#ifdef HAVE_MIC
   if( std::is_same< DestinationElement, SourceElement >::value )
   {
      Devices::MICHider<void> src_ptr;
      src_ptr.pointer=(void*)source;

      //JAKA KONSTANTA se vejde do stacku 5MB?
      if(size<MIC_STACK_VAR_LIM)
      {
         uint8_t tmp[size*sizeof(SourceElement)];

         #pragma offload target(mic) in(src_ptr,size) out(tmp)
         {
              memcpy((void*)&tmp,src_ptr.pointer,size*sizeof(SourceElement));
         }

         memcpy((void*)destination,(void*)&tmp,size*sizeof(SourceElement));
         return true;
      }
      else
      {
          //direct -- pomalejší
          uint8_t* tmp=(uint8_t*)destination;
          #pragma offload target(mic) in(src_ptr,size) out(tmp:length(size))
          {
              memcpy((void*)tmp,src_ptr.pointer,size*sizeof(SourceElement));
          }
          return true;
      }
   }
   else
   {
      Devices::MICHider<const SourceElement> src_ptr;
      src_ptr.pointer=source;

      if(size<MIC_STACK_VAR_LIM)
      {
         uint8_t tmp[size*sizeof(DestinationElement)];

         #pragma offload target(mic) in(src_ptr,size) out(tmp)
         {
              DestinationElement *dst=(DestinationElement*)&tmp;
              for(int i=0;i<size;i++)
                  dst[i]=src_ptr.pointer[i];
         }

         memcpy((void*)destination,(void*)&tmp,size*sizeof(DestinationElement));
         return true;
      }
      else
      {
          //direct pseudo heap-- pomalejší
          uint8_t* tmp=(uint8_t*)destination;
          #pragma offload target(mic) in(src_ptr,size) out(tmp:length(size*sizeof(DestinationElement)))
          {
              DestinationElement *dst=(DestinationElement*)tmp;
              for(int i=0;i<size;i++)
                  dst[i]=src_ptr.pointer[i];
          }
          return true;
      }
   }
   return false;
#else
   throw Exceptions::MICSupportMissing();
#endif
}


template< typename Element1,
          typename Element2,
          typename Index >
bool
ArrayOperations< Devices::Host, Devices::MIC >::
compareMemory( const Element1* destination,
               const Element2* source,
               const Index size )
{
   /***
    * Here, destination is on host and source is on MIC device.
    */
   TNL_ASSERT( destination, );
   TNL_ASSERT( source, );
   TNL_ASSERT( size >= 0, std::cerr << "size = " << size );
#ifdef HAVE_MIC
   Index compared( 0 );
   Index transfer( 0 );
   std::size_t max_transfer=MIC_STACK_VAR_LIM/sizeof(Element2);
   uint8_t host_buffer[max_transfer*sizeof(Element2)];

   Devices::MICHider<const Element2> src_ptr;

   while( compared < size )
   {
     transfer=min(size-compared,max_transfer);
     src_ptr.pointer=source+compared;
     #pragma offload target(mic) out(host_buffer) in(src_ptr,transfer)
     {
         memcpy((void*)&host_buffer,(void*)src_ptr.pointer,transfer*sizeof(Element2));
     }
     if( ! ArrayOperations< Devices::Host >::compareMemory( &destination[ compared ], (Element2*)&host_buffer, transfer ) )
     {
        return false;
     }
     compared += transfer;
   }
   return true;
#else
   throw Exceptions::MICSupportMissing();
#endif
}

/****
 * Operations Host -> MIC
 */
template< typename DestinationElement,
          typename SourceElement,
          typename Index >
bool
ArrayOperations< Devices::MIC, Devices::Host >::
copyMemory( DestinationElement* destination,
            const SourceElement* source,
            const Index size )
{
   TNL_ASSERT( destination, );
   TNL_ASSERT( source, );
   TNL_ASSERT( size >= 0, std::cerr << "size = " << size );
#ifdef HAVE_MIC
   if( std::is_same< DestinationElement, SourceElement >::value )
   {
      Devices::MICHider<void> dst_ptr;
      dst_ptr.pointer=(void*)destination;

      //JAKA KONSTANTA se vejde do stacku 5MB?
      if(size<MIC_STACK_VAR_LIM)
      {
         uint8_t tmp[size*sizeof(SourceElement)];
         memcpy((void*)&tmp,(void*)source,size*sizeof(SourceElement));

         #pragma offload target(mic) in(dst_ptr,tmp,size)
         {
              memcpy(dst_ptr.pointer,(void*)&tmp,size*sizeof(SourceElement));
         }

         return true;
      }
      else
      {
          //direct pseudo heap-- pomalejší
          uint8_t* tmp=(uint8_t*)source;
          #pragma offload target(mic) in(dst_ptr,size) in(tmp:length(size))
          {
              memcpy(dst_ptr.pointer,(void*)tmp,size*sizeof(SourceElement));
          }
          return true;
      }
   }
   else
   {
      Devices::MICHider<DestinationElement> dst_ptr;
      dst_ptr.pointer=destination;

      if(size<MIC_STACK_VAR_LIM)
      {
         uint8_t tmp[size*sizeof(SourceElement)];
         memcpy((void*)&tmp,(void*)source,size*sizeof(SourceElement));

         #pragma offload target(mic) in(dst_ptr,size,tmp)
         {
              SourceElement *src=(SourceElement*)&tmp;
              for(int i=0;i<size;i++)
                  dst_ptr.pointer[i]=src[i];
         }
         return true;
      }
      else
      {
          //direct pseudo heap-- pomalejší
          uint8_t* tmp=(uint8_t*)source;
          #pragma offload target(mic) in(dst_ptr,size) in(tmp:length(size*sizeof(SourceElement)))
          {
              SourceElement *src=(SourceElement*)tmp;
              for(int i=0;i<size;i++)
                  dst_ptr.pointer[i]=src[i];
          }
          return true;
      }
   }
   return false;
#else
   throw Exceptions::MICSupportMissing();
#endif
}

template< typename Element1,
          typename Element2,
          typename Index >
bool
ArrayOperations< Devices::MIC, Devices::Host >::
compareMemory( const Element1* hostData,
               const Element2* deviceData,
               const Index size )
{
   TNL_ASSERT( hostData, );
   TNL_ASSERT( deviceData, );
   TNL_ASSERT( size >= 0, std::cerr << "size = " << size );
   return ArrayOperations< Devices::Host, Devices::MIC >::compareMemory( deviceData, hostData, size );
}

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
