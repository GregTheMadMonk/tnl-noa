/***************************************************************************
                          tnlVectorOperationsTester.h  -  description
                             -------------------
    begin                : Jan 12, 2010
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef TNLASSERT_H_
#define TNLASSERT_H_

/****
 * Debugging assert
 */

#ifndef NDEBUG

#include <iostream>
#include <stdlib.h>

using namespace std;

#ifdef HAVE_CUDA
#define tnlAssert( ___tnl__assert_condition, ___tnl__assert_command )                                    \
   if( ! ( ___tnl__assert_condition ) )                                                                  \
   {                                                                                                     \
   printf( "Assertion '%s' failed !!! \n File: %s \n Line: %d \n Diagnostics: Not supported with CUDA.", \
           __STRING( ___tnl__assert_condition ),                                                         \
           __FILE__,                                                                                     \
           __LINE__ );                                                                                   \
        abort();                                                                                         \
   }
#else
#define tnlAssert( ___tnl__assert_condition, ___tnl__assert_command )                       \
	if( ! ( ___tnl__assert_condition ) )                                                     \
	{                                                                                        \
	cerr << "Assertion '" << __STRING( ___tnl__assert_condition ) << "' failed !!!" << endl  \
             << "File: " << __FILE__ << endl                                                \
             << "Function: " << __PRETTY_FUNCTION__ << endl                                 \
             << "Line: " << __LINE__ << endl                                                \
             << "Diagnostics: ";                                                            \
        ___tnl__assert_command;                                                             \
        throw EXIT_FAILURE;                                                                 \
	}
#endif /* HAVE_CUDA */
#else /* #ifndef NDEBUG */
#define tnlAssert( ___tnl__assert_condition, ___tnl__assert_command )
#endif /* #ifndef NDEBUG */

/****
 * Static assert
 */

#ifndef HAVE_CUDA // TODO: fix this when nvcc can compile it
// static_assert() available for g++ 4.3 or newer with -std=c++0x or -std=gnu++0x
#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 2)) && defined(__GXX_EXPERIMENTAL_CXX0X__)
#define CXX0X_STATIC_ASSERT_AVAILABLE
#endif


#if defined(CXX0X_STATIC_ASSERT_AVAILABLE)
#define tnlStaticAssert(expression, msg) static_assert(expression, msg)
#else

#define JOIN(X, Y)  JOIN2(X, Y)
#define JOIN2(X, Y) X##Y

// Incomplete-type implementation of compile time assertion
template<bool x> struct TNL_STATIC_ASSERTION_FAILURE;
template<>       struct TNL_STATIC_ASSERTION_FAILURE<true> { enum { value = 1 }; };

template<int x> struct static_assert_test{};


#define tnlStaticAssert(expression, msg) \
   typedef static_assert_test< sizeof( TNL_STATIC_ASSERTION_FAILURE< expression > ) >\
      JOIN(static_assertion_failure_identifier, __LINE__)

#endif // defined(CXX0X_STATIC_ASSERT_AVAILABLE)

#else
#define tnlStaticAssert(expression, msg)
#endif // ifndef HAVE_CUDA

#endif /* TNLASSERT_H_ */
