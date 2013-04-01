/***************************************************************************
                          tnlCudaVectorOperationsTest.cu  -  description
                             -------------------
    begin                : Mar 31, 2013
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
 
#include "tnlCudaVectorOperationsTester.h"
#include "../../tnlUnitTestStarter.h"
 
int main( int argc, char* argv[] )
{
   if( ! tnlUnitTestStarter :: run< tnlCudaVectorOperationsTester< double > >() )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}