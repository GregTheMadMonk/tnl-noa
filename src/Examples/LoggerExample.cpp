#include <iostream>
#include <TNL/Logger.h>
#include <TNL/Config::ParameterContainer.h>

using namespace TNL;
using namespace std;
       
int main()
{
    Logger logger(50,stream);
    
    Config::ParameterContainer parameters;
    logger.writeSystemInformation(parameters);

    logger.writeHeader("MyTitle");
    parameters.template addParameter< String >( "Device:", "cuda" );
    parameters.template addParameter< String >( "Real type:", "double" );
    parameters.template addParameter< String >( "Index type:", "int" );
    logger.writeSeparator();
    logger.writeSystemInformation(parameters);
    logger.writeSeparator();
}

