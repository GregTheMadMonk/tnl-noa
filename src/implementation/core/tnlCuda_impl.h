/***************************************************************************
                          tnlCuda_impl.h  -  description
                             -------------------
    begin                : Jul 11, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef TNLCUDA_IMPL_H_
#define TNLCUDA_IMPL_H_

inline tnlString tnlCuda :: getDeviceType()
{
   return tnlString( "tnlCuda" );
}

inline tnlDeviceEnum tnlCuda :: getDevice()
{
   return tnlCudaDevice;
};

template< typename Element, typename Index >
void tnlCuda :: allocateMemory( Element*& data, const Index size )
{
   allocateMemoryCuda( data, size );
}

template< typename Element >
void tnlCuda :: freeMemory( Element* data )
{
   freeMemoryCuda( data );
}


template< typename Element >
void tnlCuda :: setMemoryElement( Element* data,
                                         const Element& value )
{
   setMemoryCuda( data, value, 1, maxGridSize );
}

template< typename Element >
Element tnlCuda :: getMemoryElement( const Element* data )
{
   Element result;
   copyMemoryCudaToHost( &result, data, 1 );
   return result;
}

template< typename Element, typename Index >
Element& tnlCuda :: getArrayElementReference( Element* data, const Index i )
{
   tnlAssert( false, );
   abort();
}

template< typename Element, typename Index >
const Element& tnlCuda :: getArrayElementReference(const Element* data, const Index i )
{
   tnlAssert( false, );
   abort();
}

template< typename DestinationElement,
          typename SourceElement,
          typename Index,
          typename Device >
bool tnlCuda :: memcpy( DestinationElement* destination,
                        const SourceElement* source,
                        const Index size )
{
   switch( Device :: getDevice() )
   {
      case tnlHostDevice:
         return copyMemoryHostToCuda( destination, source, size );
      case tnlCudaDevice:
         return copyMemoryCudaToCuda( destination, source, size );
   }
   return true;
}


template< typename Element, typename Index, typename Device >
bool tnlCuda :: memcpy( Element* destination,
                        const Element* source,
                        const Index size )
{
   return tnlCuda :: memcpy< Element, Element, Index, Device >
                           ( destination,
                             source,
                             size );
}

template< typename Element, typename Index, typename Device >
bool tnlCuda :: memcmp( const Element* data1,
                        const Element* data2,
                        const Index size )
{
   switch( Device :: getDevice() )
   {
      case tnlHostDevice:
         return compareMemoryHostCuda( data2, data1, size );
      case tnlCudaDevice:
         return compareMemoryCuda( data1, data2, size );
   }
}

template< typename Element, typename Index >
bool tnlCuda :: memset( Element* destination,
                        const Element& value,
                        const Index size )
{
   return setMemoryCuda( destination, value, size, maxGridSize );
}

inline int tnlCuda :: getMaxGridSize()
{
   return maxGridSize;
}

inline void tnlCuda :: setMaxGridSize( int newMaxGridSize )
{
   maxGridSize = newMaxGridSize;
}

inline int tnlCuda :: getMaxBlockSize()
{
   return maxBlockSize;
}

inline void tnlCuda :: setMaxBlockSize( int newMaxBlockSize )
{
   maxBlockSize = newMaxBlockSize;
}


#endif /* TNLCUDA_IMPL_H_ */
