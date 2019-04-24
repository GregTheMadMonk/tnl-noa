/***************************************************************************
                          StaticArray.h  -  description
                             -------------------
    begin                : Feb 10, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/String.h>
#include <TNL/File.h>

namespace TNL {
namespace Containers {

/**
 * \brief Array with constant size.
 *
 * \param Size Size of static array. Number of its elements.
 * \param Value Type of the values in static array.
 */
template< int Size, typename Value >
class StaticArray
{
   public:
   using ValueType = Value;
   using IndexType = int;
   enum { size = Size };

   /**
    * \brief Gets size of this array.
    */
   __cuda_callable__
   static constexpr int getSize();

   /**
    * \brief Basic constructor.
    *
    * Constructs an empty static array.
    */
   __cuda_callable__
   inline StaticArray();

   /**
    * \brief Constructor that sets all array components (with the number of \e Size) to value \e v.
    *
    * Once this static array is constructed, its size can not be changed.
    * \tparam _unused
    * \param v[Size]
    */
   // Note: the template avoids ambiguity of overloaded functions with literal 0 and pointer
   // reference: https://stackoverflow.com/q/4610503
   template< typename _unused = void >
   __cuda_callable__
   inline StaticArray( const Value v[ Size ] );

   /**
    * \brief Constructor that sets all array components to value \e v.
    *
    * \param v Reference to a value.
    */
   __cuda_callable__
   inline StaticArray( const Value& v );

   /**
    * \brief Copy constructor.
    *
    * Constructs a copy of another static array \e v.
    */
   __cuda_callable__
   inline StaticArray( const StaticArray< Size, Value >& v );

   inline StaticArray( const std::initializer_list< Value > &elems );

   /**
    * \brief Gets type of this array.
    */
   static String getType();


   /**
    * \brief Gets all data of this static array.
    */
   __cuda_callable__
   inline Value* getData();

   /**
    * \brief Gets all data of this static array.
    */
   __cuda_callable__
   inline const Value* getData() const;

   /**
    * \brief Accesses specified element at the position \e i and returns a reference to its value.
    *
    * \param i Index position of an element.
    */
   __cuda_callable__
   inline const Value& operator[]( int i ) const;

   /**
    * \brief Accesses specified element at the position \e i and returns a reference to its value.
    *
    * \param i Index position of an element.
    */
   __cuda_callable__
   inline Value& operator[]( int i );

   /**
    * \brief Assigns another static \e array to this array, replacing its current contents.
    */
   __cuda_callable__
   inline StaticArray< Size, Value >& operator = ( const StaticArray< Size, Value >& array );

   /**
    * \brief Assigns another static \e array to this array, replacing its current contents.
    */
   template< typename Array >
   __cuda_callable__
   inline StaticArray< Size, Value >& operator = ( const Array& array );

   /**
    * \brief This function checks whether this static array is equal to another \e array.
    *
    * Return \e true if the arrays are equal in size. Otherwise returns \e false.
    */
   template< typename Array >
   __cuda_callable__
   inline bool operator == ( const Array& array ) const;

   /**
    * \brief This function checks whether this static array is not equal to another \e array.
    *
    * Return \e true if the arrays are not equal in size. Otherwise returns \e false.
    */
   template< typename Array >
   __cuda_callable__
   inline bool operator != ( const Array& array ) const;

   template< typename OtherValue >
   __cuda_callable__
   operator StaticArray< Size, OtherValue >() const;

   /**
    * \brief Sets all values of this static array to \e val.
    */
   __cuda_callable__
   inline void setValue( const ValueType& val );

   /**
    * \brief Saves this static array into the \e file.
    * \param file Reference to a file.
    */
   bool save( File& file ) const;

   /**
    * \brief Loads data from the \e file to this static array.
    * \param file Reference to a file.
    */
   bool load( File& file);

   /**
    * \brief Sorts the elements in this static array into ascending order.
    */
   void sort();

   /**
    * \brief Writes the array values into stream \e str with specified \e separator.
    *
    * @param str Reference to a stream.
    * @param separator Character separating the array values in the stream \e str.
    * Is set to " " by default.
    */
   std::ostream& write( std::ostream& str, const char* separator = " " ) const;

   protected:
   Value data[ Size ];
};

/**
 * \brief Specific static array with the size of 1. Works like the class StaticArray.
 */
template< typename Value >
class StaticArray< 1, Value >
{
   public:
   typedef Value ValueType;
   typedef int     IndexType;
   enum { size = 1 };

   /**
    * \brief Gets size of this array.
    */
   __cuda_callable__
   static constexpr int getSize();

   /** \brief See StaticArray::StaticArray().*/
   __cuda_callable__
   inline StaticArray();

   /** \brief See StaticArray::StaticArray(const Value v[Size]).*/
   // Note: the template avoids ambiguity of overloaded functions with literal 0 and pointer
   // reference: https://stackoverflow.com/q/4610503
   template< typename _unused = void >
   __cuda_callable__
   inline StaticArray( const Value v[ size ] );

   /** \brief See StaticArray::StaticArray(const Value& v).*/
   __cuda_callable__
   inline StaticArray( const Value& v );

   /** \brief See StaticArray::StaticArray( const StaticArray< Size, Value >& v ).*/
   __cuda_callable__
   inline StaticArray( const StaticArray< size, Value >& v );

   inline StaticArray( const std::initializer_list< Value > &elems );

   /** \brief See StaticArray::getType().*/
   static String getType();

   /** \brief See StaticArray::getData().*/
   __cuda_callable__
   inline Value* getData();

   /** \brief See StaticArray::getData() const.*/
   __cuda_callable__
   inline const Value* getData() const;

   /** \brief See StaticArray::operator[]( int i ) const.*/
   __cuda_callable__
   inline const Value& operator[]( int i ) const;

   /** \brief See StaticArray::operator[]( int i ).*/
   __cuda_callable__
   inline Value& operator[]( int i );

   /** \brief Returns the first coordinate - the first element of this static array.*/
   __cuda_callable__
   inline Value& x();

   /** \brief Returns the first coordinate - the first element of this static array.*/
   __cuda_callable__
   inline const Value& x() const;

   /** \brief Similar to StaticArray::operator = ( const StaticArray< Size, Value >& array ) only with Size equal to 1.*/
   __cuda_callable__
   inline StaticArray< 1, Value >& operator = ( const StaticArray< 1, Value >& array );

   /** \brief See StaticArray::operator = (const Array& array).*/
   template< typename Array >
   __cuda_callable__
   inline StaticArray< 1, Value >& operator = ( const Array& array );

   /** \brief See StaticArray::operator == (const Array& array) const.*/
   template< typename Array >
   __cuda_callable__
   inline bool operator == ( const Array& array ) const;

   /** \brief See StaticArray::operator != (const Array& array) const.*/
   template< typename Array >
   __cuda_callable__
   inline bool operator != ( const Array& array ) const;

   template< typename OtherValue >
   __cuda_callable__
   operator StaticArray< 1, OtherValue >() const;

   /** \brief See StaticArray::setValue().*/
   __cuda_callable__
   inline
   void setValue( const ValueType& val );

   /** \brief See StaticArray::save().*/
   bool save( File& file ) const;

   /** \brief See StaticArray::load().*/
   bool load( File& file);

   /** \brief See StaticArray::sort().*/
   void sort();

   /** \brief See StaticArray::write().*/
   std::ostream& write( std::ostream& str, const char* separator = " " ) const;

   protected:
   Value data[ size ];
};

/**
 * \brief Specific static array with the size of 2. Works like the class StaticArray.
 */
template< typename Value >
class StaticArray< 2, Value >
{
   public:
   typedef Value ValueType;
   typedef int     IndexType;
   enum { size = 2 };

   /**
    * \brief Gets size of this array.
    */
   __cuda_callable__
   static constexpr int getSize();

   /** \brief See StaticArray::StaticArray().*/
   __cuda_callable__
   inline StaticArray();

   /** \brief See StaticArray::StaticArray(const Value v[Size]).*/
   // Note: the template avoids ambiguity of overloaded functions with literal 0 and pointer
   // reference: https://stackoverflow.com/q/4610503
   template< typename _unused = void >
   __cuda_callable__
   inline StaticArray( const Value v[ size ] );

   /** \brief See StaticArray::StaticArray(const Value& v).*/
   __cuda_callable__
   inline StaticArray( const Value& v );

   /**
    * \brief Constructor that sets the two static array components to value \e v1 and \e v2.
    *
    * \param v1 Reference to the value of first array/vector component.
    * \param v2 Reference to the value of second array/vector component.
    */
   __cuda_callable__
   inline StaticArray( const Value& v1, const Value& v2 );

   /** \brief See StaticArray::StaticArray( const StaticArray< Size, Value >& v ).*/
   __cuda_callable__
   inline StaticArray( const StaticArray< size, Value >& v );

   inline StaticArray( const std::initializer_list< Value > &elems );

   /** \brief See StaticArray::getType().*/
   static String getType();

   /** \brief See StaticArray::getData().*/
   __cuda_callable__
   inline Value* getData();

   /** \brief See StaticArray::getData() const.*/
   __cuda_callable__
   inline const Value* getData() const;

   /** \brief See StaticArray::operator[]( int i ) const.*/
   __cuda_callable__
   inline const Value& operator[]( int i ) const;

   /** \brief See StaticArray::operator[]( int i ).*/
   __cuda_callable__
   inline Value& operator[]( int i );

   /** \brief Returns the first coordinate - the first element of this static array.*/
   __cuda_callable__
   inline Value& x();

   /** \brief Returns the first coordinate - the first element of this static array.*/
   __cuda_callable__
   inline const Value& x() const;

   /** \brief Returns the second coordinate - the second element of this static array.*/
   __cuda_callable__
   inline Value& y();

   /** \brief Returns the second coordinate - the second element of this static array.*/
   __cuda_callable__
   inline const Value& y() const;

   /** \brief Similar to StaticArray::operator = ( const StaticArray< Size, Value >& array ) only with Size equal to 2.*/
   __cuda_callable__
   inline StaticArray< 2, Value >& operator = ( const StaticArray< 2, Value >& array );

   /** \brief See StaticArray::operator = (const Array& array).*/
   template< typename Array >
   __cuda_callable__
   inline StaticArray< 2, Value >& operator = ( const Array& array );

   /** \brief See StaticArray::operator == (const Array& array) const.*/
   template< typename Array >
   __cuda_callable__
   inline bool operator == ( const Array& array ) const;

   /** \brief See StaticArray::operator != (const Array& array) const.*/
   template< typename Array >
   __cuda_callable__
   inline bool operator != ( const Array& array ) const;

   template< typename OtherValue >
   __cuda_callable__
   operator StaticArray< 2, OtherValue >() const;

   /** \brief See StaticArray::setValue().*/
   __cuda_callable__
   inline void setValue( const ValueType& val );

   /** \brief See StaticArray::save().*/
   bool save( File& file ) const;

   /** \brief See StaticArray::load().*/
   bool load( File& file);

   /** \brief See StaticArray::sort().*/
   void sort();

   /** \brief See StaticArray::write().*/
   std::ostream& write( std::ostream& str, const char* separator = " " ) const;

   protected:
   Value data[ size ];
};

/**
 * \brief Specific static array with the size of 3. Works like the class StaticArray.
 */
template< typename Value >
class StaticArray< 3, Value >
{
   public:
   typedef Value ValueType;
   typedef int     IndexType;
   enum { size = 3 };

   /**
    * \brief Gets size of this array.
    */
   __cuda_callable__
   static constexpr int getSize();

   /** \brief See StaticArray::StaticArray().*/
   __cuda_callable__
   inline StaticArray();

   /** \brief See StaticArray::StaticArray(const Value v[ size ]).*/
   // Note: the template avoids ambiguity of overloaded functions with literal 0 and pointer
   // reference: https://stackoverflow.com/q/4610503
   template< typename _unused = void >
   __cuda_callable__
   inline StaticArray( const Value v[ size ] );

   /** \brief See StaticArray::StaticArray(const Value& v).*/
   __cuda_callable__
   inline StaticArray( const Value& v );

   /**
    * \brief Constructor that sets the three array components to value \e v1 \e v2 and \e v3.
    *
    * \param v1 Reference to the value of first array/vector component.
    * \param v2 Reference to the value of second array/vector component.
    * \param v3 Reference to the value of third array/vector component.
    */
   __cuda_callable__
   inline StaticArray( const Value& v1, const Value& v2, const Value& v3 );

   /** \brief See StaticArray::StaticArray( const StaticArray< Size, Value >& v ).*/
   __cuda_callable__
   inline StaticArray( const StaticArray< size, Value >& v );

   StaticArray( const std::initializer_list< Value > &elems );

   /** \brief See StaticArray::getType().*/
   static String getType();

   /** \brief See StaticArray::getData().*/
   __cuda_callable__
   inline Value* getData();

   /** \brief See StaticArray::getData() const.*/
   __cuda_callable__
   inline const Value* getData() const;

   /** \brief See StaticArray::operator[]( int i ) const.*/
   __cuda_callable__
   inline const Value& operator[]( int i ) const;

   /** \brief See StaticArray::operator[]( int i ).*/
   __cuda_callable__
   inline Value& operator[]( int i );

   /** \brief Returns the first coordinate - the first element of this static array.*/
   __cuda_callable__
   inline Value& x();

   /** \brief Returns the first coordinate - the first element of this static array.*/
   __cuda_callable__
   inline const Value& x() const;

   /** \brief Returns the second coordinate - the second element of this static array.*/
   __cuda_callable__
   inline Value& y();

   /** \brief Returns the second coordinate - the second element of this static array.*/
   __cuda_callable__
   inline const Value& y() const;

   /** \brief Returns the third coordinate - the third element of this static array.*/
   __cuda_callable__
   inline Value& z();

   /** \brief Returns the third coordinate - the third element of this static array.*/
   __cuda_callable__
   inline const Value& z() const;

   /** \brief Similar to StaticArray::operator = ( const StaticArray< Size, Value >& array ) only with Size equal to 3.*/
   __cuda_callable__
   inline StaticArray< 3, Value >& operator = ( const StaticArray< 3, Value >& array );

   /** \brief See StaticArray::operator = (const Array& array).*/
   template< typename Array >
   __cuda_callable__
   inline StaticArray< 3, Value >& operator = ( const Array& array );

   /** \brief See StaticArray::operator == (const Array& array) const.*/
   template< typename Array >
   __cuda_callable__
   inline bool operator == ( const Array& array ) const;

   /** \brief See StaticArray::operator != (const Array& array) const.*/
   template< typename Array >
   __cuda_callable__
   inline bool operator != ( const Array& array ) const;

   template< typename OtherValue >
   __cuda_callable__
   operator StaticArray< 3, OtherValue >() const;

   /** \brief See StaticArray::setValue().*/
   __cuda_callable__
   inline void setValue( const ValueType& val );

   /** \brief See StaticArray::save().*/
   bool save( File& file ) const;

   /** \brief See StaticArray::load().*/
   bool load( File& file);

   /** \brief See StaticArray::sort().*/
   void sort();

   /** \brief See StaticArray::write().*/
   std::ostream& write( std::ostream& str, const char* separator = " " ) const;

   protected:
   Value data[ size ];
};

template< int Size, typename Value >
std::ostream& operator << ( std::ostream& str, const StaticArray< Size, Value >& a );

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/StaticArray_impl.h>
#include <TNL/Containers/StaticArray1D_impl.h>
#include <TNL/Containers/StaticArray2D_impl.h>
#include <TNL/Containers/StaticArray3D_impl.h>
