include Makefile.inc

FILES = Makefile \
        Makefile.inc \
        $(SOURCES) \
	$(HEADERS)

SUBDIRS = 

SUBDIRSCLEAN=$(addsuffix clean,$(SUBDIRS))

all: bin subdirs

.PHONY:	subdirs $(SUBDIRS)
subdirs:    $(SUBDIRS)
$(SUBDIRS):
	$(MAKE) -C $@	

bin:
	mkdir -p bin

install: all
	mkdir -p $(INSTALL_DIR)/bin
	cp bin/* $(INSTALL_DIR)/bin

.PHONY:	clean
clean:	$(SUBDIRSCLEAN) clean_curdir

clean_curdir:
	rm -f *.o
	
%clean:	%
	$(MAKE) -C $< clean

dist: clean
	tar zcvf fjfi-num1-src.tgz $(SUBDIRS) $(FILES)

$(TARGETS): % : %.o
	$(CXX) $(LDFLAGS) -o $@ $< $(LDLIBS)

$(CUDA_TARGETS): % : %.cu.o
	$(CXX) $(CUDA_LDFLAGS) -o $@ $< $(CUDA_LDLIBS)

$(SOURCES:%.cpp=%.o): %.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

$(CUDA_SOURCES:%.cu=%.cu.o): %.cu.o : %.cu
	$(CUDA_COMPILER) $(CUDA_CPPFLAGS) $(CUDA_CXXFLAGS) -c -o $@ $<