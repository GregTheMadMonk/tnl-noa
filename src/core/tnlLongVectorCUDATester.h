/***************************************************************************
 tnlLongVectorCUDATester.h  -  description
 -------------------
 begin                : Dec 27, 2009
 copyright            : (C) 2009 by Tomas Oberhuber
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

#ifndef TNLLONGVECTORCUDATESTER_H_
#define TNLLONGVECTORCUDATESTER_H_

/*
 *
 */
#include <cppunit/TestSuite.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestCaller.h>
#include <cppunit/TestCase.h>
#include <core/tnlLongVectorCUDA.h>

template< class T > class tnlLongVectorCUDATester : public CppUnit :: TestCase
{
   public:
   tnlLongVectorCUDATester(){};

   virtual
   ~tnlLongVectorCUDATester(){};

   static CppUnit :: Test* suite()
   {
      CppUnit :: TestSuite suite;
      CppUnit :: TestResult result;
      suite.addTest( new CppUnit :: TestCaller< tnlLongVectorCUDATester< float > >(
                     "testAllocation",
                     & tnlLongVectorCUDATester< float > :: testAllocation )
                   );
      suite.run( &result );
   }

   void testAllocation()
   {
      tnlLongVectorCUDA< T > cuda_vector;
      CPPUNIT_ASSERT( ! cuda_vector );

   }

};

#endif /* TNLLONGVECTORCUDATESTER_H_ */
