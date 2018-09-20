# - Try to find Template Numerical Library TNL
# Once done this will define
#  TNL_FOUND - System has LibXml2
#  TNL_INCLUDE_DIRS - The LibXml2 include directories
#  TNL_LIBRARIES - The libraries needed to use LibXml2
#  TNL_DEFINITIONS - Compiler switches required for using LibXml2

find_package(PkgConfig)
pkg_check_modules(PC_TNL QUIET tnl)
set(TNL_DEFINITIONS ${PC_TNL_CFLAGS_OTHER})

find_path(TNL_INCLUDE_DIR libxml/xpath.h
          HINTS ${PC_LIBXML_INCLUDEDIR} ${PC_LIBXML_INCLUDE_DIRS}
          PATH_SUFFIXES libxml2 )

find_library(TNL_LIBRARY NAMES xml2 libxml2
             HINTS ${PC_LIBXML_LIBDIR} ${PC_LIBXML_LIBRARY_DIRS} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set TNL_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(LibXml2  DEFAULT_MSG
                                  TNL_LIBRARY TNL_INCLUDE_DIR)

mark_as_advanced(TNL_INCLUDE_DIR TNL_LIBRARY )

set(TNL_LIBRARIES ${TNL_LIBRARY} )
set(TNL_INCLUDE_DIRS ${TNL_INCLUDE_DIR} )
