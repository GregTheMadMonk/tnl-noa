/***************************************************************************
                          tnlConstSharedArray.h  -  description
                             -------------------
    begin                : Feb 13, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLCONSTSHAREDARRAY_H_
#define TNLCONSTSHAREDARRAY_H_

#include <core/tnlObject.h>

class tnlFile;
class tnlHost;

template< typename Element, typename Device, typename Index >
class tnlArray;

template< typename Element,
          typename Device = tnlHost,
          typename Index = int >
class tnlConstSharedArray : public tnlObject
{
   public:

   typedef Element ElementType;
   typedef Device DeviceType;
   typedef Index IndexType;

   tnlConstSharedArray();

   tnlString getType() const;

   void bind( const Element* _data,
              const Index _size );

   template< typename Array >
   void bind( const Array& array );

   void swap( tnlConstSharedArray< Element, Device, Index >& array );

   void reset();

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   Index getSize() const;

   Element getElement( Index i ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const Element& operator[] ( Index i ) const;

   tnlConstSharedArray< Element, Device, Index >& operator = ( const tnlConstSharedArray< Element, Device, Index >& array );

   template< typename Array >
   tnlConstSharedArray< Element, Device, Index >& operator = ( const Array& array );

   template< typename Array >
   bool operator == ( const Array& array ) const;

   template< typename Array >
   bool operator != ( const Array& array ) const;

#ifdef HAVE_CUDA
   __device__ __host__
#endif
   const Element* getData() const;

   /****
    * Returns true if non-zero size is set.
    */
   operator bool() const;

   //! This method measures data transfers done by this vector.
   /*!
    * Every time one touches this grid touches * size * sizeof( Real ) bytes are added
    * to transfered bytes in tnlStatistics.
    */
#ifdef HAVE_NOT_CXX11
   template< typename IndexType2 >
   void touch( IndexType2 touches = 1 ) const;
#else
   template< typename IndexType2 = Index >
   void touch( IndexType2 touches = 1 ) const;
#endif

   //! Method for saving the object to a file as a binary data.
   bool save( tnlFile& file ) const;

   bool save( const tnlString& fileName ) const;

   protected:

   //!Number of allocated elements
   Index size;

   //! Pointer to allocated data
   const Element* data;
};

template< typename Element, typename Device, typename Index >
ostream& operator << ( ostream& str, const tnlConstSharedArray< Element, Device, Index >& v );

#include <implementation/core/arrays/tnlConstSharedArray_impl.h>

#endif /* TNLCONSTSHAREDARRAY_H_ */
