/***************************************************************************
                          CommonData.h  -  description
                             -------------------
    begin                : Jul 14, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber

#pragma once

#include <TNL/Config/ParameterContainer.h>

namespace TNL {
namespace Problems {

class CommonData
{
   public:
   
      bool setup( const Config::ParameterContainer& parameters )
      {
         return true;
      }
};

} // namespace Problems
} // namespace TNL
