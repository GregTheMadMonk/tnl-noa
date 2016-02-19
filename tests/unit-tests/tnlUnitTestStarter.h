/***************************************************************************
                          tnlUnitTestStarter.h  -  description
                             -------------------
    begin                : Mar 30, 2013
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

#ifndef TNLUNITTESTSTARTER_H_
#define TNLUNITTESTSTARTER_H_

#include <tnlConfig.h>

#ifdef HAVE_CPPUNIT
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/CompilerOutputter.h>
#endif

#include <iostream>

class tnlUnitTestStarter
{
   public:

   template< typename Tester >
   static bool run()
   {
#ifdef HAVE_CPPUNIT
      CppUnit::TextTestRunner runner;
      runner.addTest( Tester :: suite() );
      runner.setOutputter( new CppUnit::CompilerOutputter(&runner.result(), std::cout) );
      if( ! runner.run() )
         return false;
      return true;
#else
      std :: cerr << "Error: CPPUNIT is missing." << std :: endl;
      return false;
#endif
   }
};

#endif /* TNLUNITTESTSTARTER_H_ */
