/***************************************************************************
                          tnlArray.h -  description
                             -------------------
    begin                : Jul 4, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
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

#ifndef TNLARRAY_H_
#define TNLARRAY_H_

#include <core/tnlObject.h>
#include <core/arrays/tnlSharedArray.h>

class tnlFile;
class tnlHost;

template< typename Element, typename Device, typename Index >
class tnlSharedArray;

/****
 * Array handles memory allocation and sharing of the same data between more Arrays.
 * 
 */
template< typename Element,
          typename Device = tnlHost,
          typename Index = int >
class tnlArray : public virtual tnlObject
{
   public:

      typedef Element ElementType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef tnlArray< Element, tnlHost, Index > HostType;
      typedef tnlArray< Element, tnlCuda, Index > CudaType;
      typedef tnlArray< Element, Device, Index > ThisType;
      
      tnlArray();
      
      tnlArray( const IndexType& size );
      
      tnlArray( Element* data,
                const IndexType& size );

      tnlArray( tnlArray< Element, Device, Index >& array,
                const IndexType& begin = 0,
                const IndexType& size = 0 );

      static tnlString getType();

      tnlString getTypeVirtual() const;

      static tnlString getSerializationType();

      virtual tnlString getSerializationTypeVirtual() const;

      /****
       * This sets size of the array. If the array shares data with other arrays
       * these data are released. If the current data are not shared and the current
       * size is the same as the new one, nothing happens.
       */
      bool setSize( Index size );

      template< typename Array >
      bool setLike( const Array& array );

      void bind( Element* _data,
                 const Index _size );

      void bind( const tnlArray< Element, Device, Index >& array,
                 const IndexType& begin = 0,
                 const IndexType& size = 0 );

      template< int Size >
      void bind( tnlStaticArray< Size, Element >& array );

      void swap( tnlArray< Element, Device, Index >& array );

      void reset();

      __cuda_callable__ Index getSize() const;

      void setElement( const Index& i, const Element& x );

      Element getElement( const Index& i ) const;

      __cuda_callable__ inline Element& operator[] ( const Index& i );

      __cuda_callable__ inline const Element& operator[] ( const Index& i ) const;

      tnlArray< Element, Device, Index >& operator = ( const tnlArray< Element, Device, Index >& array );

      template< typename Array >
      tnlArray< Element, Device, Index >& operator = ( const Array& array );

      template< typename Array >
      bool operator == ( const Array& array ) const;

      template< typename Array >
      bool operator != ( const Array& array ) const;

      void setValue( const Element& e );

      __cuda_callable__ const Element* getData() const;

      __cuda_callable__ Element* getData();

      /*!
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

      //! Method for loading the object from a file as a binary data.
      bool load( tnlFile& file );
      
      //! This method loads data without reallocation. 
      /****
       * This is useful for loading data into shared arrays.
       * If the array was not initialize yet, common load is
       * performed. Otherwise, the array size must fit with
       * the size of array being loaded.
       */
      bool boundLoad( tnlFile& file );
      
      bool boundLoad( const tnlString& fileName );
      
      using tnlObject::load;

      using tnlObject::save;

      ~tnlArray();

   protected:
      
      void releaseData() const;

      //!Number of elements in array
      mutable Index size;

      //! Pointer to data
      mutable Element* data;

      /****
       * Pointer to the originally allocated data. They might differ if one 
       * long array is partitioned into more shorter arrays. Each of them
       * must know the pointer on allocated data because the last one must
       * deallocate the array. If outer data (not allocated by TNL) are bind
       * then this pointer is zero since no deallocation is necessary.
       */
      mutable Element* allocationPointer;

      /****
       * Counter of objects sharing this array or some parts of it. The reference counter is
       * allocated after first sharing of the data between more arrays. This is to avoid
       * unnecessary dynamic memory allocation.
       */
      mutable int* referenceCounter;
};

template< typename Element, typename Device, typename Index >
ostream& operator << ( ostream& str, const tnlArray< Element, Device, Index >& v );

#include <core/arrays/tnlArray_impl.h>
#ifdef HAVE_MIC
    //MIC specializaton of Araray
    #include <core/arrays/tnlArrayMIC_impl.h>
#endif
#endif /* TNLARRAY_H_ */
