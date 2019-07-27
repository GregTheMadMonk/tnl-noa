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
#include <TNL/TypeTraits.h>

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
   StaticArray( const Value v[ Size ] );

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
    * \brief Constructor that sets components of arrays with Size = 2.
    *
    * \param v1 Value of the first array component.
    * \param v2 Value of the second array component.
    */
   __cuda_callable__
   inline StaticArray( const Value& v1, const Value& v2 );

   /**
    * \brief Constructor that sets components of arrays with Size = 3.
    *
    * \param v1 Value of the first array component.
    * \param v2 Value of the second array component.
    * \param v3 Value of the third array component.
    */
   __cuda_callable__
   inline StaticArray( const Value& v1, const Value& v2, const Value& v3 );

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

   /** \brief Returns the first coordinate.*/
   __cuda_callable__
   inline Value& x();

   /** \brief Returns the first coordinate.*/
   __cuda_callable__
   inline const Value& x() const;

   /** \brief Returns the second coordinate for arrays with Size >= 2.*/
   __cuda_callable__
   inline Value& y();

   /** \brief Returns the second coordinate for arrays with Size >= 2.*/
   __cuda_callable__
   inline const Value& y() const;

   /** \brief Returns the third coordinate for arrays with Size >= 3..*/
   __cuda_callable__
   inline Value& z();

   /** \brief Returns the third coordinate for arrays with Size >= 3..*/
   __cuda_callable__
   inline const Value& z() const;

   /**
    * \brief Assigns another static \e array to this array, replacing its current contents.
    */
   __cuda_callable__
   inline StaticArray< Size, Value >& operator=( const StaticArray< Size, Value >& array );

   /**
    * \brief Assigns another static \e array to this array, replacing its current contents.
    */
   template< typename Array >
   __cuda_callable__
   inline StaticArray< Size, Value >& operator=( const Array& array );

   /**
    * \brief This function checks whether this static array is equal to another \e array.
    *
    * Return \e true if the arrays are equal in size. Otherwise returns \e false.
    */
   template< typename Array >
   __cuda_callable__
   inline bool operator==( const Array& array ) const;

   /**
    * \brief This function checks whether this static array is not equal to another \e array.
    *
    * Return \e true if the arrays are not equal in size. Otherwise returns \e false.
    */
   template< typename Array >
   __cuda_callable__
   inline bool operator!=( const Array& array ) const;

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

template< int Size, typename Value >
std::ostream& operator<<( std::ostream& str, const StaticArray< Size, Value >& a );

} // namespace Containers

template< int Size, typename Value_ >
struct IsStatic< Containers::StaticArray< Size, Value_ > >
{
   static constexpr bool Value = true;
};

} // namespace TNL

#include <TNL/Containers/StaticArray.hpp>
