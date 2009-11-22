/***************************************************************************
 tnlStringTester.cpp  -  description
 -------------------
 begin                : Nov 22, 2009
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

#include <cstring>
#include "tnlStringTester.h"

void tnlStringTester :: Test( tnlTester& tester )
{
   tester. StartNewUnit( "tnlString" );

   tester. StartNewTest( "empty string" );
   tnlString empty_string;
   tester. StartNewTest( "re-typing to bool" );
   if( empty_string )
      tester. FinishTest( tnlTestFAIL );
   else tester. FinishTest( tnlTestPASS );
   tester. StartNewTest( "length of empty string" );
   if( empty_string.Length() != 0 )
      tester. FinishTest( tnlTestFAIL );
   else tester. FinishTest( tnlTestPASS );
   tester. StartNewTest( "data of empty string" );
   if( strcmp( empty_string. Data(), "" ) != 0 )
      tester. FinishTest( tnlTestFAIL );
   else tester. FinishTest( tnlTestPASS );

   tester. FinishUnit();






    /*if( object_tester. GetType() == "tnlObjectTester" )
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
       else tester. FinishTest( tnlTestFAIL );
    }*/
}
