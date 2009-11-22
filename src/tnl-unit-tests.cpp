/***************************************************************************
                          tnl-unit-tests.cpp  -  description
                             -------------------
    begin                : Nov 21, 2009
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

#include <stdlib.h>
#include <core/tnlTester.h>
#include <core/tnlStringTester.h>
#include <core/tnlObjectTester.h>

int main( int argc, char* argv[] )
{
   tnlTester tester;

   /* Testing tnlString
    *
    */
   tnlStringTester string_tester;
   string_tester. Test( tester );

   /* Testing tnlObject
    *
    */
   tnlObjectTester tnl_object_tester;
   tnl_object_tester. Test( tester );


   tester. PrintStatistics();

   return EXIT_SUCCESS;
}
