/***************************************************************************
                          MultiArray.h  -  description
                             -------------------
    begin                : Nov 25, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Assert.h>

namespace TNL {
namespace Containers {   

template< int Dimensions, typename Element = double, typename Device = Devices::Host, typename Index = int >
class MultiArray : public Array< Element, Device, Index >
{
};

template< typename Element, typename Device, typename Index >
class MultiArray< 1, Element, Device, Index > : public Array< Element, Device, Index >
{
   public:
   enum { Dimensions = 1};
   typedef Element ElementType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef MultiArray< 1, Element, Devices::Host, Index > HostType;
   typedef MultiArray< 1, Element, Devices::Cuda, Index > CudaType;


   MultiArray();

   static String getType();

   String getTypeVirtual() const;

   static String getSerializationType();

   virtual String getSerializationTypeVirtual() const;

   bool setDimensions( const Index iSize );

   bool setDimensions( const Containers::StaticVector< 1, Index >& dimensions );

   __cuda_callable__ void getDimensions( Index& iSize ) const;

   __cuda_callable__ const Containers::StaticVector< 1, Index >& getDimensions() const;

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


   template< typename MultiArrayT >
   bool operator == ( const MultiArrayT& array ) const;

   template< typename MultiArrayT >
   bool operator != ( const MultiArrayT& array ) const;

   MultiArray< 1, Element, Device, Index >& operator = ( const MultiArray< 1, Element, Device, Index >& array );

   template< typename MultiArrayT >
   MultiArray< 1, Element, Device, Index >& operator = ( const MultiArrayT& array );

   //! Method for saving the object to a file as a binary data
   bool save( File& file ) const;

   //! Method for restoring the object from a file
   bool load( File& file );

   bool save( const String& fileName ) const;

   bool load( const String& fileName );

   protected:

   Containers::StaticVector< 1, Index > dimensions;
};

template< typename Element, typename Device, typename Index >
class MultiArray< 2, Element, Device, Index > : public Array< Element, Device, Index >
{
   public:
   enum { Dimensions = 2 };
   typedef Element ElementType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef MultiArray< 2, Element, Devices::Host, Index > HostType;
   typedef MultiArray< 2, Element, Devices::Cuda, Index > CudaType;


   MultiArray();

   static String getType();

   String getTypeVirtual() const;

   static String getSerializationType();

   virtual String getSerializationTypeVirtual() const;

   bool setDimensions( const Index jSize, const Index iSize );

   bool setDimensions( const Containers::StaticVector< 2, Index >& dimensions );

   __cuda_callable__ void getDimensions( Index& jSize, Index& iSize ) const;

   __cuda_callable__ const Containers::StaticVector< 2, Index >& getDimensions() const;

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

   template< typename MultiArrayT >
   bool operator == ( const MultiArrayT& array ) const;

   template< typename MultiArrayT >
   bool operator != ( const MultiArrayT& array ) const;

   MultiArray< 2, Element, Device, Index >& operator = ( const MultiArray< 2, Element, Device, Index >& array );

   template< typename MultiArrayT >
   MultiArray< 2, Element, Device, Index >& operator = ( const MultiArrayT& array );

   //! Method for saving the object to a file as a binary data
   bool save( File& file ) const;

   //! Method for restoring the object from a file
   bool load( File& file );

   bool save( const String& fileName ) const;

   bool load( const String& fileName );

   protected:

   Containers::StaticVector< 2, Index > dimensions;
};

template< typename Element, typename Device, typename Index >
class MultiArray< 3, Element, Device, Index > : public Array< Element, Device, Index >
{
   public:

   enum { Dimensions = 3 };
   typedef Element ElementType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef MultiArray< 3, Element, Devices::Host, Index > HostType;
   typedef MultiArray< 3, Element, Devices::Cuda, Index > CudaType;


   MultiArray();

   static String getType();

   String getTypeVirtual() const;

   static String getSerializationType();

   virtual String getSerializationTypeVirtual() const;

   bool setDimensions( const Index k, const Index j, const Index iSize );

   bool setDimensions( const Containers::StaticVector< 3, Index >& dimensions );

   __cuda_callable__ void getDimensions( Index& k, Index& j, Index& iSize ) const;

   __cuda_callable__ const Containers::StaticVector< 3, Index >& getDimensions() const;

   //! Set dimensions of the array using another array as a template
   template< typename MultiArrayT >
   bool setLike( const MultiArrayT& v );

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

   template< typename MultiArrayT >
   bool operator == ( const MultiArrayT& array ) const;

   template< typename MultiArrayT >
   bool operator != ( const MultiArrayT& array ) const;

   MultiArray< 3, Element, Device, Index >& operator = ( const MultiArray< 3, Element, Device, Index >& array );

   template< typename MultiArrayT >
   MultiArray< 3, Element, Device, Index >& operator = ( const MultiArrayT& array );

   //! Method for saving the object to a file as a binary data
   bool save( File& file ) const;

   //! Method for restoring the object from a file
   bool load( File& file );

   bool save( const String& fileName ) const;

   bool load( const String& fileName );

   protected:

   Containers::StaticVector< 3, Index > dimensions;
};

template< typename Element, typename Device, typename Index >
class MultiArray< 4, Element, Device, Index > : public Array< Element, Device, Index >
{
   public:

   enum { Dimensions = 4 };
   typedef Element ElementType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef MultiArray< 4, Element, Devices::Host, Index > HostType;
   typedef MultiArray< 4, Element, Devices::Cuda, Index > CudaType;


   MultiArray();

   static String getType();

   String getTypeVirtual() const;

   static String getSerializationType();

   virtual String getSerializationTypeVirtual() const;

   bool setDimensions( const Index l, const Index k, const Index j, const Index iSize );

   bool setDimensions( const Containers::StaticVector< 4, Index >& dimensions );

   __cuda_callable__ void getDimensions( Index& l, Index& k, Index& j, Index& iSize ) const;

   __cuda_callable__ const Containers::StaticVector< 4, Index >& getDimensions() const;

   //! Set dimensions of the array using another array as a template
   template< typename MultiArrayT >
   bool setLike( const MultiArrayT& v );

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

   template< typename MultiArrayT >
   bool operator == ( const MultiArrayT& array ) const;

   template< typename MultiArrayT >
   bool operator != ( const MultiArrayT& array ) const;

   MultiArray< 4, Element, Device, Index >& operator = ( const MultiArray< 4, Element, Device, Index >& array );

   template< typename MultiArrayT >
   MultiArray< 4, Element, Device, Index >& operator = ( const MultiArrayT& array );

   //! Method for saving the object to a file as a binary data
   bool save( File& file ) const;

   //! Method for restoring the object from a file
   bool load( File& file );

   bool save( const String& fileName ) const;

   bool load( const String& fileName );

   protected:

   Containers::StaticVector< 4, Index > dimensions;
};

template< typename Element, typename device, typename Index >
std::ostream& operator << ( std::ostream& str, const MultiArray< 1, Element, device, Index >& array );

template< typename Element, typename device, typename Index >
std::ostream& operator << ( std::ostream& str, const MultiArray< 2, Element, device, Index >& array );

template< typename Element, typename device, typename Index >
std::ostream& operator << ( std::ostream& str, const MultiArray< 3, Element, device, Index >& array );

template< typename Element, typename device, typename Index >
std::ostream& operator << ( std::ostream& str, const MultiArray< 4, Element, device, Index >& array );

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/MultiArray1D_impl.h>
#include <TNL/Containers/MultiArray2D_impl.h>
#include <TNL/Containers/MultiArray3D_impl.h>
#include <TNL/Containers/MultiArray4D_impl.h>

namespace TNL {
namespace Containers {
   
#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef INSTANTIATE_FLOAT
extern template class MultiArray< 1, float,  Devices::Host, int >;
#endif
extern template class MultiArray< 1, double, Devices::Host, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class MultiArray< 1, float,  Devices::Host, long int >;
#endif
extern template class MultiArray< 1, double, Devices::Host, long int >;
#endif

#ifdef INSTANTIATE_FLOAT
extern template class MultiArray< 2, float,  Devices::Host, int >;
#endif
extern template class MultiArray< 2, double, Devices::Host, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class MultiArray< 2, float,  Devices::Host, long int >;
#endif
extern template class MultiArray< 2, double, Devices::Host, long int >;
#endif

#ifdef INSTANTIATE_FLOAT
extern template class MultiArray< 3, float,  Devices::Host, int >;
#endif
extern template class MultiArray< 3, double, Devices::Host, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class MultiArray< 3, float,  Devices::Host, long int >;
#endif
extern template class MultiArray< 3, double, Devices::Host, long int >;
#endif

#ifdef INSTANTIATE_FLOAT
extern template class MultiArray< 4, float,  Devices::Host, int >;
#endif
extern template class MultiArray< 4, double, Devices::Host, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class MultiArray< 4, float,  Devices::Host, long int >;
#endif
extern template class MultiArray< 4, double, Devices::Host, long int >;
#endif

// TODO: There are problems with nvlink - it might be better in later versions
/*
#ifdef INSTANTIATE_FLOAT
extern template class MultiArray< 1, float,  Devices::Cuda, int >;
#endif
extern template class MultiArray< 1, double, Devices::Cuda, int >;
#ifdef INSTANTIATE_FLOAT
extern template class MultiArray< 1, float,  Devices::Cuda, long int >;
#endif
extern template class MultiArray< 1, double, Devices::Cuda, long int >;
#ifdef INSTANTIATE_FLOAT
extern template class MultiArray< 2, float,  Devices::Cuda, int >;
#endif
extern template class MultiArray< 2, double, Devices::Cuda, int >;
#ifdef INSTANTIATE_FLOAT
extern template class MultiArray< 2, float,  Devices::Cuda, long int >;
#endif
extern template class MultiArray< 2, double, Devices::Cuda, long int >;
#ifdef INSTANTIATE_FLOAT
extern template class MultiArray< 3, float,  Devices::Cuda, int >;
#endif
extern template class MultiArray< 3, double, Devices::Cuda, int >;
#ifdef INSTANTIATE_FLOAT
extern template class MultiArray< 3, float,  Devices::Cuda, long int >;
#endif
extern template class MultiArray< 3, double, Devices::Cuda, long int >;
#ifdef INSTANTIATE_FLOAT
extern template class MultiArray< 4, float,  Devices::Cuda, int >;
#endif
extern template class MultiArray< 4, double, Devices::Cuda, int >;
#ifdef INSTANTIATE_FLOAT
extern template class MultiArray< 4, float,  Devices::Cuda, long int >;
#endif
extern template class MultiArray< 4, double, Devices::Cuda, long int >;*/

#endif

} // namespace Containers
} // namespace TNL
