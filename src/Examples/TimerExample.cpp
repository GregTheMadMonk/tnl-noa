#include <iostream>
#include <TNL/Timer.h>

using namespace TNL;
using namespace std;
       
int main()
{
    Timer time;
    time.start();
    time.stop();
    time.getRealTime();
    time.reset();
}

