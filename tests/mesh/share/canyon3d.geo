# ABL canyon
algebraic3d

solid domain = orthobrick(0, 0, 0; 15, 8, 8);

solid box1 = orthobrick(4, -0.1, -0.1; 5, 1, 8.1) -maxh=0.2;
solid box2 = orthobrick(6, -0.1, -0.1; 7, 1, 8.1) -maxh=0.2;

solid canyon = domain and not box1 and not box2 -maxh=2;

tlo canyon;
