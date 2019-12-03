/***************************************************************************
                          SmartPointer.h  -  description
                             -------------------
    begin                : May 30, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

/**
 * \brief Namespace for TNL pointers.
 *
 * Pointers in TNL are similar to STL pointers but they work across different device.
 */
namespace Pointers {

class SmartPointer
{
   public:

      virtual bool synchronize() = 0;

};

} // namespace Pointers
} // namespace TNL
