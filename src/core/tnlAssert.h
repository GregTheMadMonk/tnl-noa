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
#include <assert.h>

//using namespace std;

#ifdef HAVE_CUDA
#define tnlAssert( ___tnl__assert_condition, ___tnl__assert_command )                                    \
   if( ! ( ___tnl__assert_condition ) )                                                                  \
   {                                                                                                     \
   printf( "Assertion '%s' failed !!! \n File: %s \n Line: %d \n Diagnostics: Not supported with CUDA.", \
           __STRING( ___tnl__assert_condition ),                                                         \
           __FILE__,                                                                                     \
           __LINE__ );                                                                                   \
                                                              \
   }

#else // HAVE_CUDA
#define tnlAssert( ___tnl__assert_condition, ___tnl__assert_command )                       \
	if( ! ( ___tnl__assert_condition ) )                                                     \
	{                                                                                        \
	std::cerr << "Assertion '" << __STRING( ___tnl__assert_condition ) << "' failed !!!" << endl  \
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


#endif /* TNLASSERT_H_ */
