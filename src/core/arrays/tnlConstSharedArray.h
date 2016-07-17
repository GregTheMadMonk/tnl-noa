/***************************************************************************
                          tnlConstSharedArray.h  -  description
                             -------------------
    begin                : Feb 13, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <tnlObject.h>

namespace TNL {

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
   typedef tnlConstSharedArray< Element, tnlHost, Index > HostType;
   typedef tnlConstSharedArray< Element, tnlCuda, Index > CudaType;

   tnlConstSharedArray();

   static tnlString getType();

   tnlString getTypeVirtual() const;

   static tnlString getSerializationType();

   virtual tnlString getSerializationTypeVirtual() const;

   void bind( const Element* _data,
              const Index _size );

   template< typename Array >
   void bind( const Array& array,
              IndexType index = 0,
              IndexType size = 0 );

   void swap( tnlConstSharedArray< Element, Device, Index >& array );

   void reset();

   __cuda_callable__ Index getSize() const;

   Element getElement( Index i ) const;

   __cuda_callable__ const Element& operator[] ( Index i ) const;

   tnlConstSharedArray< Element, Device, Index >& operator = ( const tnlConstSharedArray< Element, Device, Index >& array );

   template< typename Array >
   tnlConstSharedArray< Element, Device, Index >& operator = ( const Array& array );

   template< typename Array >
   bool operator == ( const Array& array ) const;

   template< typename Array >
   bool operator != ( const Array& array ) const;

   __cuda_callable__ const Element* getData() const;

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

} // namespace TNL

#include <core/arrays/tnlConstSharedArray_impl.h>


