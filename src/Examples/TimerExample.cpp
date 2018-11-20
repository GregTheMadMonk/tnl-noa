#include <iostream>
#include <TNL/Timer.h>
#include <unistd.h>

using namespace TNL;
using namespace std;
       
int main()
{
    unsigned int microseconds = 0.5;
    Timer time;
    time.start();
    usleep(microseconds);
    time.stop();
    time.getRealTime();
    time.reset();
}

