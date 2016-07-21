/***************************************************************************
                          tnlDebugEntry.h  -  description
                             -------------------
    begin                : 2005/08/16
    copyright            : (C) 2005 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef tnlDebugEntryH
#define tnlDebugEntryH

#include <string>

using namespace std;

//! This structure desrcibes which classes and methods to debug and which one not
struct tnlDebugEntry
{
   //! Method or function name
   /*! If it is realy function the method name is empty
    */
   string function_name;

   //! It is true if we are debuging this function/method
   bool debug;

   //! Flag for interactive debuging
   bool interactive;

   void Print() {};
};


#endif
