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

#include <core/tnlArray.h>
#include <core/tnlTuple.h>
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

   tnlMultiArray();

   tnlMultiArray( const tnlString& name );

   tnlString getType() const;

   bool setDimensions( const Index iSize );

   bool setDimensions( const tnlTuple< 1, Index >& dimensions );

   void getDimensions( Index& iSize ) const;

   const tnlTuple< 1, Index >& getDimensions() const;

   //! Set dimensions of the array using another array as a template
   template< typename MultiArray >
   bool setLike( const MultiArray& v );
   
   Index getElementIndex( const Index i ) const;

   void setElement( const Index i, Element value );

   //! This method can be used for general access to the elements of the arrays.
   /*! It does not return reference but value. So it can be used to access
    *  arrays in different adress space (usualy GPU device).
    *  See also operator().
    */
   Element getElement( const Index i ) const;

   //! Operator for accessing elements of the array.
   /*! It returns reference to given elements so it cannot be
    *  used to access elements of arrays in different adress space
    *  (GPU device usualy).
    */
   Element& operator()( const Index i );

   const Element& operator()( const Index i ) const;

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

   protected:

   tnlTuple< 1, Index > dimensions;
};

template< typename Element, typename Device, typename Index >
class tnlMultiArray< 2, Element, Device, Index > : public tnlArray< Element, Device, Index >
{
   public:
   enum { Dimensions = 2 };
   typedef Element ElementType;
   typedef Device DeviceType;
   typedef Index IndexType;

   tnlMultiArray();

   tnlMultiArray( const tnlString& name );

   tnlString getType() const;

   bool setDimensions( const Index jSize, const Index iSize );

   bool setDimensions( const tnlTuple< 2, Index >& dimensions );

   void getDimensions( Index& jSize, Index& iSize ) const;

   const tnlTuple< 2, Index >& getDimensions() const;

   //! Set dimensions of the array using another array as a template
   template< typename MultiArray >
   bool setLike( const MultiArray& v );

   Index getElementIndex( const Index j, const Index i ) const;

   void setElement( const Index j, const Index i, Element value );

   //! This method can be used for general access to the elements of the arrays.
   /*! It does not return reference but value. So it can be used to access
    *  arrays in different adress space (usualy GPU device).
    *  See also operator().
    */
   Element getElement( const Index j, const Index i ) const;

   //! Operator for accessing elements of the array.
   /*! It returns reference to given elements so it cannot be
    *  used to access elements of arrays in different adress space
    *  (GPU device usualy).
    */
   Element& operator()( const Index j, const Index i );

   const Element& operator()( const Index j, const Index i ) const;

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

   protected:

   tnlTuple< 2, Index > dimensions;
};

template< typename Element, typename Device, typename Index >
class tnlMultiArray< 3, Element, Device, Index > : public tnlArray< Element, Device, Index >
{
   public:

   enum { Dimensions = 3 };
   typedef Element ElementType;
   typedef Device DeviceType;
   typedef Index IndexType;

   tnlMultiArray();

   tnlMultiArray( const tnlString& name );

   tnlString getType() const;

   bool setDimensions( const Index k, const Index j, const Index iSize );

   bool setDimensions( const tnlTuple< 3, Index >& dimensions );

   void getDimensions( Index& k, Index& j, Index& iSize ) const;

   const tnlTuple< 3, Index >& getDimensions() const;

   //! Set dimensions of the array using another array as a template
   template< typename MultiArray >
   bool setLike( const MultiArray& v );

   Index getElementIndex( const Index k, const Index j, const Index i ) const;

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
   Element& operator()( const Index k, const Index j, const Index i );

   const Element& operator()( const Index k, const Index j, const Index i ) const;

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

   protected:

   tnlTuple< 3, Index > dimensions;
};

template< typename Element, typename Device, typename Index >
class tnlMultiArray< 4, Element, Device, Index > : public tnlArray< Element, Device, Index >
{
   public:

   enum { Dimensions = 4 };
   typedef Element ElementType;
   typedef Device DeviceType;
   typedef Index IndexType;

   tnlMultiArray();

   tnlMultiArray( const tnlString& name );

   tnlString getType() const;

   bool setDimensions( const Index l, const Index k, const Index j, const Index iSize );

   bool setDimensions( const tnlTuple< 4, Index >& dimensions );

   void getDimensions( Index& l, Index& k, Index& j, Index& iSize ) const;

   const tnlTuple< 4, Index >& getDimensions() const;

   //! Set dimensions of the array using another array as a template
   template< typename MultiArray >
   bool setLike( const MultiArray& v );

   Index getElementIndex( const Index l, const Index k, const Index j, const Index i ) const;

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
   Element& operator()( const Index l, const Index k, const Index j, const Index i );

   const Element& operator()( const Index l, const Index k, const Index j, const Index i ) const;

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

   protected:

   tnlTuple< 4, Index > dimensions;
};

template< typename Element, typename device, typename Index >
ostream& operator << ( ostream& str, const tnlMultiArray< 1, Element, device, Index >& array );

template< typename Element, typename device, typename Index >
ostream& operator << ( ostream& str, const tnlMultiArray< 2, Element, device, Index >& array );

template< typename Element, typename device, typename Index >
ostream& operator << ( ostream& str, const tnlMultiArray< 3, Element, device, Index >& array );

template< typename Element, typename device, typename Index >
ostream& operator << ( ostream& str, const tnlMultiArray< 4, Element, device, Index >& array );


#include <core/implementation/tnlMultiArray1D_impl.h>
#include <core/implementation/tnlMultiArray2D_impl.h>
#include <core/implementation/tnlMultiArray3D_impl.h>
#include <core/implementation/tnlMultiArray4D_impl.h>

#endif /* TNLMULTIARRAY_H_ */
