/***************************************************************************
                          VectorOperationsTester.h  -  description
                             -------------------
    begin                : Jan 12, 2010
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

/****
 * Debugging assert
 */

#ifndef NDEBUG

#include <iostream>
#include <stdlib.h>
#include <assert.h>

#endif

namespace TNL {

#ifndef NDEBUG   
   
#ifdef HAVE_CUDA
#define TNL_ASSERT( ___tnl__assert_condition, ___tnl__assert_command )                                    \
   if( ! ( ___tnl__assert_condition ) )                                                                  \
   {                                                                                                     \
   printf( "Assertion '%s' failed !!! \n File: %s \n Line: %d \n Diagnostics: Not supported with CUDA.\n", \
           __STRING( ___tnl__assert_condition ),                                                         \
           __FILE__,                                                                                     \
           __LINE__ );                                                                                   \
                                                              \
   }

#else // HAVE_CUDA
#define TNL_ASSERT( ___tnl__assert_condition, ___tnl__assert_command )                       \
	if( ! ( ___tnl__assert_condition ) )                                                     \
	{                                                                                        \
	std::cerr << "Assertion '" << __STRING( ___tnl__assert_condition ) << "' failed !!!" << std::endl  \
             << "File: " << __FILE__ << std::endl                                                \
             << "Function: " << __PRETTY_FUNCTION__ << std::endl                                 \
             << "Line: " << __LINE__ << std::endl                                                \
             << "Diagnostics: ";                                                            \
        ___tnl__assert_command;                                                             \
        throw EXIT_FAILURE;                                                                 \
	}
#endif /* HAVE_CUDA */
#else /* #ifndef NDEBUG */
#define TNL_ASSERT( ___tnl__assert_condition, ___tnl__assert_command )
#endif /* #ifndef NDEBUG */

} // namespace TNL
