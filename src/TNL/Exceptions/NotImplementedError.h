// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <stdexcept>

namespace TNL {
namespace Exceptions {

struct NotImplementedError : public std::runtime_error
{
   NotImplementedError( std::string msg = "Something is not implemented." ) : std::runtime_error( msg ) {}
};

}  // namespace Exceptions
}  // namespace TNL
