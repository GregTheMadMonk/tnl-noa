/*
 * tnlAssert.h
 *
 *  Created on: Jan 12, 2010
 *      Author: oberhuber
 */

#ifndef TNLASSERT_H_

#ifdef DEBUG

#include <iostream>
#include <stdlib.h>

using namespace std;

#define tnlAssert( ___tnl__assert_condition, ___tnl__assert_command )     \
	if( ! ___tnl__assert_condition )                                  \
	{                                                                 \
	cerr << "Assertion '___tnl__assert_condition' failed !!!" << endl \
             << "File: __FILE__" << endl                                  \
             << "Function: __PRETTY_FUNCTION__ " << endl                  \
             << "Line: __LINE__" << endl                                  \
             << "Diagnostics: ";                                          \
        ___tnl__assert_command;                                           \
        exit( EXIT_FAILURE );                                             \
	}
#else
#define tnlAssert( ___tnl__assert_condition, ___tnl__assert_command )
#endif

#define TNLASSERT_H_


#endif /* TNLASSERT_H_ */
