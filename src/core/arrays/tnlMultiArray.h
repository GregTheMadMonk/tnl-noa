/***************************************************************************
                          tnlMultiArray.h  -  description
                             -------------------
    begin                : Nov 25, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
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

#ifndef TNLMULTIARRAY_H_
#define TNLMULTIARRAY_H_

#include <iostream>
#include <core/arrays/tnlArray.h>
#include <core/vectors/tnlStaticVector.h>
#include <core/tnlAssert.h>


template< int Dimensions, typename Element = double, typename Device = tnlHost, typename Index = int >
class tnlMultiArray : public tnlArray< Element, Device, Index >
{
};

template< typename Element, typename Device, typename Index >
class tnlMultiArray< 1, Element, Device, Index > : public tnlArray< Element, Device, Index >
{
   public:
   enum { Dimensions = 1};
   typedef Element ElementType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlMultiArray< 1, Element, tnlHost, Index > HostType;
   typedef tnlMultiArray< 1, Element, tnlCuda, Index > CudaType;


   tnlMultiArray();

   tnlMultiArray( const tnlString& name );

   static tnlString getType();

   tnlString getTypeVirtual() const;

   static tnlString getSerializationType();

   virtual tnlString getSerializationTypeVirtual() const;

   bool setDimensions( const Index iSize );

   bool setDimensions( const tnlStaticVector< 1, Index >& dimensions );

   __cuda_callable__ void getDimensions( Index& iSize ) const;

   __cuda_callable__ const tnlStaticVector< 1, Index >& getDimensions() const;

   //! Set dimensions of the array using another array as a template
   template< typename MultiArray >
   bool setLike( const MultiArray& v );
   
   void reset();

   __cuda_callable__ Index getElementIndex( const Index i ) const;

   void setElement( const Index i, Element value );

   //! This method can be used for general access to the elements of the arrays.
   /*! It does not return reference but value. So it can be used to access
    *  arrays in different address space (usually GPU device).
    *  See also operator().
    */
   Element getElement( const Index i ) const;

   //! Operator for accessing elements of the array.
   __cuda_callable__ Element& operator()( const Index i );

   __cuda_callable__ const Element& operator()( const Index i ) const;


   template< typename MultiArray >
   bool operator == ( const MultiArray& array ) const;

   template< typename MultiArray >
   bool operator != ( const MultiArray& array ) const;

   tnlMultiArray< 1, Element, Device, Index >& operator = ( const tnlMultiArray< 1, Element, Device, Index >& array );

   template< typename MultiArray >
   tnlMultiArray< 1, Element, Device, Index >& operator = ( const MultiArray& array );

   //! Method for saving the object to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   protected:

   tnlStaticVector< 1, Index > dimensions;
};

template< typename Element, typename Device, typename Index >
class tnlMultiArray< 2, Element, Device, Index > : public tnlArray< Element, Device, Index >
{
   public:
   enum { Dimensions = 2 };
   typedef Element ElementType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlMultiArray< 2, Element, tnlHost, Index > HostType;
   typedef tnlMultiArray< 2, Element, tnlCuda, Index > CudaType;


   tnlMultiArray();

   tnlMultiArray( const tnlString& name );

   static tnlString getType();

   tnlString getTypeVirtual() const;

   static tnlString getSerializationType();

   virtual tnlString getSerializationTypeVirtual() const;

   bool setDimensions( const Index jSize, const Index iSize );

   bool setDimensions( const tnlStaticVector< 2, Index >& dimensions );

   __cuda_callable__ void getDimensions( Index& jSize, Index& iSize ) const;

   __cuda_callable__ const tnlStaticVector< 2, Index >& getDimensions() const;

   //! Set dimensions of the array using another array as a template
   template< typename MultiArray >
   bool setLike( const MultiArray& v );

   void reset();

   __cuda_callable__ Index getElementIndex( const Index j, const Index i ) const;

   void setElement( const Index j, const Index i, Element value );

   //! This method can be used for general access to the elements of the arrays.
   /*! It does not return reference but value. So it can be used to access
    *  arrays in different adress space (usualy GPU device).
    *  See also operator().
    */
   Element getElement( const Index j, const Index i ) const;

   //! Operator for accessing elements of the array.
   /*! It returns reference to given elements so it cannot be
    *  used to access elements of arrays in different address space
    *  (GPU device usually).
    */
   __cuda_callable__ Element& operator()( const Index j, const Index i );

   __cuda_callable__ const Element& operator()( const Index j, const Index i ) const;

   template< typename MultiArray >
   bool operator == ( const MultiArray& array ) const;

   template< typename MultiArray >
   bool operator != ( const MultiArray& array ) const;

   tnlMultiArray< 2, Element, Device, Index >& operator = ( const tnlMultiArray< 2, Element, Device, Index >& array );

   template< typename MultiArray >
   tnlMultiArray< 2, Element, Device, Index >& operator = ( const MultiArray& array );

   //! Method for saving the object to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   protected:

   tnlStaticVector< 2, Index > dimensions;
};

template< typename Element, typename Device, typename Index >
class tnlMultiArray< 3, Element, Device, Index > : public tnlArray< Element, Device, Index >
{
   public:

   enum { Dimensions = 3 };
   typedef Element ElementType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlMultiArray< 3, Element, tnlHost, Index > HostType;
   typedef tnlMultiArray< 3, Element, tnlCuda, Index > CudaType;


   tnlMultiArray();

   tnlMultiArray( const tnlString& name );

   static tnlString getType();

   tnlString getTypeVirtual() const;

   static tnlString getSerializationType();

   virtual tnlString getSerializationTypeVirtual() const;

   bool setDimensions( const Index k, const Index j, const Index iSize );

   bool setDimensions( const tnlStaticVector< 3, Index >& dimensions );

   __cuda_callable__ void getDimensions( Index& k, Index& j, Index& iSize ) const;

   __cuda_callable__ const tnlStaticVector< 3, Index >& getDimensions() const;

   //! Set dimensions of the array using another array as a template
   template< typename MultiArray >
   bool setLike( const MultiArray& v );

   void reset();

   __cuda_callable__ Index getElementIndex( const Index k, const Index j, const Index i ) const;

   void setElement( const Index k, const Index j, const Index i, Element value );

   //! This method can be used for general access to the elements of the arrays.
   /*! It does not return reference but value. So it can be used to access
    *  arrays in different adress space (usualy GPU device).
    *  See also operator().
    */
   Element getElement( const Index k, const Index j, const Index i ) const;

   //! Operator for accessing elements of the array.
   /*! It returns reference to given elements so it cannot be
    *  used to access elements of arrays in different adress space
    *  (GPU device usualy).
    */
   __cuda_callable__ Element& operator()( const Index k, const Index j, const Index i );

   __cuda_callable__ const Element& operator()( const Index k, const Index j, const Index i ) const;

   template< typename MultiArray >
   bool operator == ( const MultiArray& array ) const;

   template< typename MultiArray >
   bool operator != ( const MultiArray& array ) const;

   tnlMultiArray< 3, Element, Device, Index >& operator = ( const tnlMultiArray< 3, Element, Device, Index >& array );

   template< typename MultiArray >
   tnlMultiArray< 3, Element, Device, Index >& operator = ( const MultiArray& array );

   //! Method for saving the object to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   protected:

   tnlStaticVector< 3, Index > dimensions;
};

template< typename Element, typename Device, typename Index >
class tnlMultiArray< 4, Element, Device, Index > : public tnlArray< Element, Device, Index >
{
   public:

   enum { Dimensions = 4 };
   typedef Element ElementType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlMultiArray< 4, Element, tnlHost, Index > HostType;
   typedef tnlMultiArray< 4, Element, tnlCuda, Index > CudaType;


   tnlMultiArray();

   tnlMultiArray( const tnlString& name );

   static tnlString getType();

   tnlString getTypeVirtual() const;

   static tnlString getSerializationType();

   virtual tnlString getSerializationTypeVirtual() const;

   bool setDimensions( const Index l, const Index k, const Index j, const Index iSize );

   bool setDimensions( const tnlStaticVector< 4, Index >& dimensions );

   __cuda_callable__ void getDimensions( Index& l, Index& k, Index& j, Index& iSize ) const;

   __cuda_callable__ const tnlStaticVector< 4, Index >& getDimensions() const;

   //! Set dimensions of the array using another array as a template
   template< typename MultiArray >
   bool setLike( const MultiArray& v );

   void reset();

   __cuda_callable__ Index getElementIndex( const Index l, const Index k, const Index j, const Index i ) const;

   void setElement( const Index l, const Index k, const Index j, const Index i, Element value );

   //! This method can be used for general access to the elements of the arrays.
   /*! It does not return reference but value. So it can be used to access
    *  arrays in different adress space (usualy GPU device).
    *  See also operator().
    */
   Element getElement( const Index l, const Index k, const Index j, const Index i ) const;

   //! Operator for accessing elements of the array.
   /*! It returns reference to given elements so it cannot be
    *  used to access elements of arrays in different adress space
    *  (GPU device usualy).
    */
   __cuda_callable__ Element& operator()( const Index l, const Index k, const Index j, const Index i );

   __cuda_callable__ const Element& operator()( const Index l, const Index k, const Index j, const Index i ) const;

   template< typename MultiArray >
   bool operator == ( const MultiArray& array ) const;

   template< typename MultiArray >
   bool operator != ( const MultiArray& array ) const;

   tnlMultiArray< 4, Element, Device, Index >& operator = ( const tnlMultiArray< 4, Element, Device, Index >& array );

   template< typename MultiArray >
   tnlMultiArray< 4, Element, Device, Index >& operator = ( const MultiArray& array );

   //! Method for saving the object to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   protected:

   tnlStaticVector< 4, Index > dimensions;
};

template< typename Element, typename device, typename Index >
ostream& operator << ( ostream& str, const tnlMultiArray< 1, Element, device, Index >& array );

template< typename Element, typename device, typename Index >
ostream& operator << ( ostream& str, const tnlMultiArray< 2, Element, device, Index >& array );

template< typename Element, typename device, typename Index >
ostream& operator << ( ostream& str, const tnlMultiArray< 3, Element, device, Index >& array );

template< typename Element, typename device, typename Index >
ostream& operator << ( ostream& str, const tnlMultiArray< 4, Element, device, Index >& array );


#include <core/arrays/tnlMultiArray1D_impl.h>
#include <core/arrays/tnlMultiArray2D_impl.h>
#include <core/arrays/tnlMultiArray3D_impl.h>
#include <core/arrays/tnlMultiArray4D_impl.h>

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiArray< 1, float,  tnlHost, int >;
#endif
extern template class tnlMultiArray< 1, double, tnlHost, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiArray< 1, float,  tnlHost, long int >;
#endif
extern template class tnlMultiArray< 1, double, tnlHost, long int >;
#endif

#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiArray< 2, float,  tnlHost, int >;
#endif
extern template class tnlMultiArray< 2, double, tnlHost, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiArray< 2, float,  tnlHost, long int >;
#endif
extern template class tnlMultiArray< 2, double, tnlHost, long int >;
#endif

#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiArray< 3, float,  tnlHost, int >;
#endif
extern template class tnlMultiArray< 3, double, tnlHost, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiArray< 3, float,  tnlHost, long int >;
#endif
extern template class tnlMultiArray< 3, double, tnlHost, long int >;
#endif

#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiArray< 4, float,  tnlHost, int >;
#endif
extern template class tnlMultiArray< 4, double, tnlHost, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiArray< 4, float,  tnlHost, long int >;
#endif
extern template class tnlMultiArray< 4, double, tnlHost, long int >;
#endif

// TODO: There are problems with nvlink - it might be better in later versions
/*
#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiArray< 1, float,  tnlCuda, int >;
#endif
extern template class tnlMultiArray< 1, double, tnlCuda, int >;
#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiArray< 1, float,  tnlCuda, long int >;
#endif
extern template class tnlMultiArray< 1, double, tnlCuda, long int >;
#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiArray< 2, float,  tnlCuda, int >;
#endif
extern template class tnlMultiArray< 2, double, tnlCuda, int >;
#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiArray< 2, float,  tnlCuda, long int >;
#endif
extern template class tnlMultiArray< 2, double, tnlCuda, long int >;
#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiArray< 3, float,  tnlCuda, int >;
#endif
extern template class tnlMultiArray< 3, double, tnlCuda, int >;
#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiArray< 3, float,  tnlCuda, long int >;
#endif
extern template class tnlMultiArray< 3, double, tnlCuda, long int >;
#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiArray< 4, float,  tnlCuda, int >;
#endif
extern template class tnlMultiArray< 4, double, tnlCuda, int >;
#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiArray< 4, float,  tnlCuda, long int >;
#endif
extern template class tnlMultiArray< 4, double, tnlCuda, long int >;*/

#endif

#endif /* TNLMULTIARRAY_H_ */
