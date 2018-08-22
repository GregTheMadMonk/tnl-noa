/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

/***************************************************************************
                          UniquePointerTest.cpp  -  description
                             -------------------
    begin                : May 28, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

#include <cstdlib>
#include <TNL/Devices/Host.h>
#include <TNL/Pointers/UniquePointer.h>
#include <TNL/Containers/StaticArray.h>

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>
#endif

using namespace TNL;

#ifdef HAVE_GTEST 
TEST( UniquePointerTest, ConstructorTest )
{
   typedef TNL::Containers::StaticArray< 2, int  > TestType;
   UniquePointer< TestType, Devices::Host > ptr1;

   ptr1->x() = 0;
   ptr1->y() = 0;
   ASSERT_EQ( ptr1->x(), 0 );
   ASSERT_EQ( ptr1->y(), 0 );

   UniquePointer< TestType, Devices::Host > ptr2( 1, 2 );
   ASSERT_EQ( ptr2->x(), 1 );
   ASSERT_EQ( ptr2->y(), 2 );

   ptr1 = ptr2;
   ASSERT_EQ( ptr1->x(), 1 );
   ASSERT_EQ( ptr1->y(), 2 );
};
#endif

#include "../GtestMissingError.h"
int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
   return RUN_ALL_TESTS();
#else
   throw GtestMissingError();
#endif
}
