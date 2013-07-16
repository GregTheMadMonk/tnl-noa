/***************************************************************************
                          tnlArrayOperationsTest.cu  -  description
                             -------------------
    begin                : Jul 16, 2013
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

#include "tnlArrayOperationsTester.h"
#include "../../tnlUnitTestStarter.h"
#include <core/tnlCuda.h>

int main( int argc, char* argv[] )
{
   if( ! tnlUnitTestStarter :: run< tnlArrayOperationsTester< int, tnlCuda > >() )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}