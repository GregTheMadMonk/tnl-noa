/***************************************************************************
                          terminal-colors.h  -  description
                             -------------------
    begin                : Feb 7, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

const tnlString red( "\033[0;31m" );
const tnlString green( "\033[1;32m" );
const tnlString yellow( "\033[1;33m" );
const tnlString cyan( "\033[0;36m" );
const tnlString magenta( "\033[0;35m" );
const tnlString bold();
const tnlString reset( "\033[0m" );

} // namespace TNL

