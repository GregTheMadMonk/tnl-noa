INSTALL( FILES tnl-quickstart
               tnl-quickstart.py
         DESTINATION bin
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )

INSTALL( FILES Makefile.in
               main.h.in
               problem.h.in
               problem_impl.h.in
         DESTINATION share
         PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ )