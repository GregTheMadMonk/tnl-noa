/***************************************************************************
                          tnlObjectTester.cpp  -  description
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

#include "tnlObjectTester.h"

void tnlObjectTester :: Test( tnlTester& tester )
{
   tester. StartNewUnit( "tnlObject" );
   tnlObjectTester object_tester;

   tester. StartNewTest( "whether tnlObject returns correct type" );
   if( object_tester. GetType() == "tnlObjectTester" )
      tester. FinishTest( tnlTestPASS );
   else tester. FinishTest( tnlTestFAIL );

   tester. StartNewTest( "whether tnlObject correctly set name" );
   const char* name = "tnlObject-unit-test";
   object_tester. SetName( name );
   if( object_tester. GetName() != name )
      tester. FinishTest( tnlTestFAIL );
   else
   {
      object_tester. SetName( "" );
      if( object_tester. GetName() != "" )
         tester. FinishTest( tnlTestFAIL );
      else tester. FinishTest( tnlTestPASS );
   }
   tester. FinishUnit();


}
