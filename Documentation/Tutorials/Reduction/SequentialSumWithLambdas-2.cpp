   auto volatileReduce = [] (volatile double& x, const volatile double& y) { x += y; };
