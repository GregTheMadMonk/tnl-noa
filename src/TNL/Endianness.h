/***************************************************************************
                          Endianness.h  -  description
                             -------------------
    begin                : Mar 11, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <climits>
#include <type_traits>

namespace TNL {

/**
 * \brief Function takes a value and swaps its endianness.
 *
 * Reference: https://stackoverflow.com/a/4956493
 */
template< typename T >
T swapEndianness(T u)
{
   static_assert( CHAR_BIT == 8, "CHAR_BIT != 8" );
   static_assert( std::is_fundamental< T >::value, "swap_endian works only for fundamental types" );

   union
   {
      T u;
      unsigned char u8[sizeof(T)];
   } source, dest;

   source.u = u;

   for (std::size_t k = 0; k < sizeof(T); k++)
      dest.u8[k] = source.u8[sizeof(T) - k - 1];

   return dest.u;
}

/**
 * \brief Function returns `true` iff the system executing the program is little endian.
 */
inline bool
isLittleEndian()
{
   const unsigned int tmp1 = 1;
   const unsigned char *tmp2 = reinterpret_cast<const unsigned char*>(&tmp1);
   if (*tmp2 != 0)
      return true;
   return false;
}

/**
 * \brief Function takes a value and returns its big endian representation.
 */
template< typename T >
T forceBigEndian(T value)
{
   static bool swap = isLittleEndian();
   if( swap )
      return swapEndianness(value);
   return value;
}

}
