/***************************************************************************
                          tnlSolverTester.h  -  description
                             -------------------
    begin                : Mar 17, 2013
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

#ifndef TNLSOLVERTEST_H_
#define TNLSOLVERTEST_H_

#include <tnlConfig.h>
#include <iostream>

using namespace std;

#ifdef HAVE_CPPUNIT

#include <cppunit/ui/text/TestRunner.h>
#include <iostream>
#include "tnlSolverTester.h"

using namespace std;

int main( int argc, char* argv[] )
{
   CppUnit :: TextTestRunner runner;
   runner. addTest( tnlSolverTester :: suite() );

}
#else
int main( int argc, char* argv[] )
{
   cerr << "UNIT TESTS ARE DISABLED." << endl;
   return 0;
}
#endif


#endif /* TNLSOLVERTEST_H_ */