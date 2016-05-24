
# These variables substituted by configure
TRUNK    = /Users/wsttiger/src/m-a-d-n-e-s-s
CXX      = mpicxx
CXXFLAGS =  -std=c++11 -g -O0 -Wall -diag-disable remark,279,654,1125  -ip -no-prec-div -mkl=sequential -ansi -xHOST 
CPPFLAGS =  -DMPICH_SKIP_MPICXX=1 -DOMPI_SKIP_MPICXX=1 -I$(TRUNK)/include -I$(TRUNK)/src -I$(TRUNK)/src/apps
LDFLAGS  = -Wl,-no_pie 
LIBS     = -ltbb  -L/opt/intel/composer_xe_2015.1.108/mkl/lib -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm  

# Directories holding libraries
LIBWORLDDIR=$(TRUNK)/src/madness/world/.libs/
LIBTENSORDIR=$(TRUNK)/src/madness/tensor/.libs/
LIBMISCDIR=$(TRUNK)/src/madness/misc/.libs/
LIBMRADIR=$(TRUNK)/src/madness/mra/.libs/
LIBTINYXMLDIR=$(TRUNK)/src/madness/external/tinyxml/.libs/
LIBMUPARSERDIR=$(TRUNK)/src/madness/external/muParser/.libs/

# Individual libraries
LIBWORLD=$(LIBWORLDDIR)/libMADworld.a
LIBTENSOR=$(LIBTENSORDIR)/libMADtensor.a
LIBLINALG=$(LIBTENSORDIR)/libMADlinalg.a
LIBMISC=$(LIBMISCDIR)/libMADmisc.a
LIBMRA=$(LIBMRADIR)/libMADmra.a
LIBTINYXML=$(LIBTINYXMLDIR)/libMADtinyxml.a
LIBMUPARSER=$(LIBMUPARSERDIR)/libMADmuparser.a



# Most scientific/numeric applications will link against these libraries
MRALIBS=$(LIBMRA) $(LIBLINALG) $(LIBTENSOR) $(LIBMISC) $(LIBMUPARSER) \
        $(LIBTINYXML) $(LIBWORLD) 

# This to enable implicit Gnumake rule for linking from single source
LDLIBS := $(MRALIBS) $(LIBS)


# Define your targets below here ... this is just an example
.PHONY: clean

OBJ_TF = test_function.o

scott:   $(OBJ_TF)
	$(CXX) -o $@ $^ $(LDLIBS)

OBJ_DF = dataflow2.o

flow:   $(OBJ_DF)
	$(CXX) -o $@ $^ $(LDLIBS)

.PHONY: clean

clean:
	rm -f $(binaries) *.o
