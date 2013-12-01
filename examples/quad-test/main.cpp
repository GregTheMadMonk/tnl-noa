/***************************************************************************
								  main.cpp  -  description
									  -------------------
	 begin					 : Jan 12, 2013
	 copyright				: (C) 2013 by Tomas Oberhuber
	 email					 : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *																			*
 *	This program is free software; you can redistribute it and/or modify	*
 *	it under the terms of the GNU General Public License as published by	*
 *	the Free Software Foundation; either version 2 of the License, or		*
 *	(at your option) any later version.										*
 *																			*
 ***************************************************************************/

#include "quad-test-conf.h"
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>
#include <matrices/tnlCSRMatrix.h>
//#include "../../src/matrix/tnlCSRMatrix.h"
#include "Quadcpp.h"

int main(int argc, char* argv[]) {
	tnlParameterContainer parameters;
	tnlConfigDescription conf_desc;
	if(conf_desc.ParseConfigDescription(CONFIG_FILE) != 0)
		return EXIT_FAILURE;
	if(!ParseCommandLine(argc, argv, conf_desc, parameters)) {
		conf_desc.PrintUsage(argv[ 0 ]);
		return EXIT_FAILURE;
	}

	tnlString inputFile = parameters.GetParameter <tnlString> ("input-file");
	tnlFile binaryFile;
	if(! binaryFile.open(inputFile, tnlReadMode)) {
		cerr << "I am not able to open the file " << inputFile << "." << endl;
		return 1;
	}
	tnlCSRMatrix <double> doubleMatrix("double");
	if(! doubleMatrix.load(binaryFile)) {
		cerr << "Unable to restore the CSR matrix." << endl;
		return 1;
	}
	binaryFile.close();
	
	tnlCSRMatrix <QuadDouble> quadMatrix("quad");
	quadMatrix = doubleMatrix;
	return EXIT_SUCCESS;
}