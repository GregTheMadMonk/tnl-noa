/***************************************************************************
                          MultiVector.h  -  description
                             -------------------
    begin                : Nov 25, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/Vectors/Vector.h>
#include <TNL/Vectors/StaticVector.h>
#include <TNL/Assert.h>

namespace TNL {
namespace Vectors {   
   
template< int Dimensions, typename Real = double, typename Device = tnlHost, typename Index = int >
class MultiVector : public Vector< Real, Device, Index >
{
};

template< typename Real, typename Device, typename Index >
class MultiVector< 1, Real, Device, Index > : public Vector< Real, Device, Index >
{
   public:
   enum { Dimensions = 1};
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef MultiVector< Dimensions, Real, tnlHost, Index > HostType;
   typedef MultiVector< Dimensions, Real, tnlCuda, Index > CudaType;

   MultiVector();

   MultiVector( const String& name );

   static String getType();

   String getTypeVirtual() const;

   static String getSerializationType();

   virtual String getSerializationTypeVirtual() const;

   bool setDimensions( const Index iSize );

   bool setDimensions( const StaticVector< Dimensions, Index >& dimensions );

   void getDimensions( Index& iSize ) const;

   const StaticVector< 1, Index >& getDimensions() const;

   //! Set dimensions of the Vector using another Vector as a template
   template< typename MultiVector >
   bool setLike( const MultiVector& v );
 
   Index getElementIndex( const Index i ) const;

   void setElement( const Index i, Real value );

   //! This method can be used for general access to the elements of the Vectors.
   /*! It does not return reference but value. So it can be used to access
    *  Vectors in different adress space (usualy GPU device).
    *  See also operator().
    */
   Real getElement( const Index i ) const;

   //! Operator for accessing elements of the Vector.
   /*! It returns reference to given elements so it cannot be
    *  used to access elements of Vectors in different adress space
    *  (GPU device usualy).
    */
   Real& operator()( const Index i );

   const Real& operator()( const Index i ) const;

   template< typename MultiVector >
   bool operator == ( const MultiVector& Vector ) const;

   template< typename MultiVector >
   bool operator != ( const MultiVector& Vector ) const;

   MultiVector< 1, Real, Device, Index >& operator = ( const MultiVector< 1, Real, Device, Index >& Vector );

   template< typename MultiVectorT >
   MultiVector< 1, Real, Device, Index >& operator = ( const MultiVectorT& Vector );

   //! Method for saving the object to a file as a binary data
   bool save( File& file ) const;

   //! Method for restoring the object from a file
   bool load( File& file );

   bool save( const String& fileName ) const;

   bool load( const String& fileName );

   protected:

   StaticVector< Dimensions, Index > dimensions;
};

template< typename Real, typename Device, typename Index >
class MultiVector< 2, Real, Device, Index > : public Vector< Real, Device, Index >
{
   public:
   enum { Dimensions = 2 };
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef MultiVector< Dimensions, Real, tnlHost, Index > HostType;
   typedef MultiVector< Dimensions, Real, tnlCuda, Index > CudaType;

   MultiVector();

   MultiVector( const String& name );

   static String getType();

   String getTypeVirtual() const;

   static String getSerializationType();

   virtual String getSerializationTypeVirtual() const;

   bool setDimensions( const Index jSize, const Index iSize );

   bool setDimensions( const StaticVector< 2, Index >& dimensions );

   void getDimensions( Index& jSize, Index& iSize ) const;

   const StaticVector< 2, Index >& getDimensions() const;

   //! Set dimensions of the Vector using another Vector as a template
   template< typename MultiVector >
   bool setLike( const MultiVector& v );

   Index getElementIndex( const Index j, const Index i ) const;

   void setElement( const Index j, const Index i, Real value );

   //! This method can be used for general access to the elements of the Vectors.
   /*! It does not return reference but value. So it can be used to access
    *  Vectors in different adress space (usualy GPU device).
    *  See also operator().
    */
   Real getElement( const Index j, const Index i ) const;

   //! Operator for accessing elements of the Vector.
   /*! It returns reference to given elements so it cannot be
    *  used to access elements of Vectors in different adress space
    *  (GPU device usualy).
    */
   Real& operator()( const Index j, const Index i );

   const Real& operator()( const Index j, const Index i ) const;

   template< typename MultiVector >
   bool operator == ( const MultiVector& Vector ) const;

   template< typename MultiVector >
   bool operator != ( const MultiVector& Vector ) const;

   MultiVector< 2, Real, Device, Index >& operator = ( const MultiVector< 2, Real, Device, Index >& Vector );

   template< typename MultiVectorT >
   MultiVector< 2, Real, Device, Index >& operator = ( const MultiVectorT& Vector );

   //! Method for saving the object to a file as a binary data
   bool save( File& file ) const;

   //! Method for restoring the object from a file
   bool load( File& file );

   bool save( const String& fileName ) const;

   bool load( const String& fileName );

   protected:

   StaticVector< 2, Index > dimensions;
};

template< typename Real, typename Device, typename Index >
class MultiVector< 3, Real, Device, Index > : public Vector< Real, Device, Index >
{
   public:

   enum { Dimensions = 3 };
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef MultiVector< Dimensions, Real, tnlHost, Index > HostType;
   typedef MultiVector< Dimensions, Real, tnlCuda, Index > CudaType;

   MultiVector();

   MultiVector( const String& name );

   static String getType();

   String getTypeVirtual() const;

   static String getSerializationType();

   virtual String getSerializationTypeVirtual() const;

   bool setDimensions( const Index k, const Index j, const Index iSize );

   bool setDimensions( const StaticVector< 3, Index >& dimensions );

   void getDimensions( Index& k, Index& j, Index& iSize ) const;

   const StaticVector< 3, Index >& getDimensions() const;

   //! Set dimensions of the Vector using another Vector as a template
   template< typename MultiVector >
   bool setLike( const MultiVector& v );

   Index getElementIndex( const Index k, const Index j, const Index i ) const;

   void setElement( const Index k, const Index j, const Index i, Real value );

   //! This method can be used for general access to the elements of the Vectors.
   /*! It does not return reference but value. So it can be used to access
    *  Vectors in different adress space (usualy GPU device).
    *  See also operator().
    */
   Real getElement( const Index k, const Index j, const Index i ) const;

   //! Operator for accessing elements of the Vector.
   /*! It returns reference to given elements so it cannot be
    *  used to access elements of Vectors in different adress space
    *  (GPU device usualy).
    */
   Real& operator()( const Index k, const Index j, const Index i );

   const Real& operator()( const Index k, const Index j, const Index i ) const;

   template< typename MultiVector >
   bool operator == ( const MultiVector& Vector ) const;

   template< typename MultiVector >
   bool operator != ( const MultiVector& Vector ) const;

   MultiVector< 3, Real, Device, Index >& operator = ( const MultiVector< 3, Real, Device, Index >& Vector );

   template< typename MultiVectorT >
   MultiVector< 3, Real, Device, Index >& operator = ( const MultiVectorT& Vector );

   //! Method for saving the object to a file as a binary data
   bool save( File& file ) const;

   //! Method for restoring the object from a file
   bool load( File& file );

   bool save( const String& fileName ) const;

   bool load( const String& fileName );

   protected:

   StaticVector< 3, Index > dimensions;
};

template< typename Real, typename Device, typename Index >
class MultiVector< 4, Real, Device, Index > : public Vector< Real, Device, Index >
{
   public:

   enum { Dimensions = 4 };
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef MultiVector< Dimensions, Real, tnlHost, Index > HostType;
   typedef MultiVector< Dimensions, Real, tnlCuda, Index > CudaType;

   MultiVector();

   MultiVector( const String& name );

   static String getType();

   String getTypeVirtual() const;

   static String getSerializationType();

   virtual String getSerializationTypeVirtual() const;

   bool setDimensions( const Index l, const Index k, const Index j, const Index iSize );

   bool setDimensions( const StaticVector< 4, Index >& dimensions );

   void getDimensions( Index& l, Index& k, Index& j, Index& iSize ) const;

   const StaticVector< 4, Index >& getDimensions() const;

   //! Set dimensions of the Vector using another Vector as a template
   template< typename MultiVector >
   bool setLike( const MultiVector& v );

   Index getElementIndex( const Index l, const Index k, const Index j, const Index i ) const;

   void setElement( const Index l, const Index k, const Index j, const Index i, Real value );

   //! This method can be used for general access to the elements of the Vectors.
   /*! It does not return reference but value. So it can be used to access
    *  Vectors in different adress space (usualy GPU device).
    *  See also operator().
    */
   Real getElement( const Index l, const Index k, const Index j, const Index i ) const;

   //! Operator for accessing elements of the Vector.
   /*! It returns reference to given elements so it cannot be
    *  used to access elements of Vectors in different adress space
    *  (GPU device usualy).
    */
   Real& operator()( const Index l, const Index k, const Index j, const Index i );

   const Real& operator()( const Index l, const Index k, const Index j, const Index i ) const;

   template< typename MultiVector >
   bool operator == ( const MultiVector& Vector ) const;

   template< typename MultiVector >
   bool operator != ( const MultiVector& Vector ) const;

   MultiVector< 4, Real, Device, Index >& operator = ( const MultiVector< 4, Real, Device, Index >& Vector );

   template< typename MultiVectorT >
   MultiVector< 4, Real, Device, Index >& operator = ( const MultiVectorT& Vector );

   //! Method for saving the object to a file as a binary data
   bool save( File& file ) const;

   //! Method for restoring the object from a file
   bool load( File& file );

   bool save( const String& fileName ) const;

   bool load( const String& fileName );

   protected:

   StaticVector< 4, Index > dimensions;
};

template< typename Real, typename device, typename Index >
std::ostream& operator << ( std::ostream& str, const MultiVector< 1, Real, device, Index >& Vector );

template< typename Real, typename device, typename Index >
std::ostream& operator << ( std::ostream& str, const MultiVector< 2, Real, device, Index >& Vector );

template< typename Real, typename device, typename Index >
std::ostream& operator << ( std::ostream& str, const MultiVector< 3, Real, device, Index >& Vector );

template< typename Real, typename device, typename Index >
std::ostream& operator << ( std::ostream& str, const MultiVector< 4, Real, device, Index >& Vector );

} // namespace Vectors
} // namespace TNL

#include <TNL/Vectors/MultiVector1D_impl.h>
#include <TNL/Vectors/MultiVector2D_impl.h>
#include <TNL/Vectors/MultiVector3D_impl.h>
#include <TNL/Vectors/MultiVector4D_impl.h>
