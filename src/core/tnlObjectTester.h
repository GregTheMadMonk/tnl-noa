/***************************************************************************
                          tnlObjectTester.h  -  description
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

#ifndef TNLOBJECTTESTER_H_
#define TNLOBJECTTESTER_H_

#include <core/tnlObject.h>
#include <core/tnlTester.h>

class tnlObjectTester : public tnlObject
{
   public:

   //! Type getter
   tnlString GetType() const
   {
      return tnlString( "tnlObjectTester" );
   }

   void Test( tnlTester& tester );
};


#endif /* TNLOBJECTTESTER_H_ */
