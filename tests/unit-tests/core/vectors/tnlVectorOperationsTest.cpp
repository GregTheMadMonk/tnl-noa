/***************************************************************************
                          tnlVectorOperationsTest.cpp  -  description
                             -------------------
    begin                : Jul 15, 2013
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

#include "tnlVectorOperationsTester.h"
#include "../../tnlUnitTestStarter.h"
#include <core/tnlHost.h>

int main( int argc, char* argv[] )
{
   if( ! tnlUnitTestStarter :: run< tnlVectorOperationsTester< double, tnlHost > >() )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}



