/***************************************************************************
 tnlStringTester.h  -  description
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

#ifndef TNLSTRINGTESTER_H_
#define TNLSTRINGTESTER_H_

#include <core/tnlString.h>
#include <core/tnlTester.h>

/*
 *
 */
class tnlStringTester : public tnlString
{
   public:

   void Test( tnlTester& tester );


};

#endif /* TNLSTRINGTESTER_H_ */
