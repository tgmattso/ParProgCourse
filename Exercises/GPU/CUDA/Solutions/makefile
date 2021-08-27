#
#  USAGE:
#     make          ... to build the program
#     make test     ... to run the default test case
#     make clean    ... remove object and executable files.
#

# verify that you are including the right make.def file for the platform
include ../make.def

COMMON      = ../Cpp_common

EXES     = vadd$(EXE) matmul$(EXE)

MMULOBJS = matmul.$(OBJ) matrix_lib.$(OBJ)

COMM_OBJS = $(COMMON)/wtime.$(OBJ) 

HEADERS_DEPS   = cuda_util.hpp	matmul.hpp	matrix_lib.hpp ./Cpp_common/cl.hpp ./Cpp_common/util.hpp

all: $(EXES)

vadd$(EXE): vadd.$(OBJ)  $(COMM_OBJS) vadd.cl 
	$(CLINKER) $(CFLAGS) $(OPENCLFLAGS) -o vadd$(EXE) vadd.$(OBJ) \
                         $(COMM_OBJS) $(LIBS)

matmul$(EXE): $(MMULOBJS) $(COMM_OBJS) 
	$(CLINKER) $(CFLAGS) $(OPENCLFLAGS) -o matmul$(EXE) $(MMULOBJS) \
                         $(COMM_OBJS) $(LIBS)

$(COMMON)/wtime.$(OBJ): $(COMMON)/wtime.c
	$(CC) -c $(CFLAGS) -o $(COMMON)/wtime.$(OBJ)  $(COMMON)/wtime.c

test: $(EXES)
	$(PRE)vadd$(EXE); \
        $(PRE)matmul$(EXE)

clean:
	$(RM) $(EXES) *.$(OBJ)

veryclean:
	$(RM) $(EXES) $(COMM_OBJS) *.$(OBJ)

.SUFFIXES:
.SUFFIXES: .c .cpp .cu .$(OBJ)

.c.$(OBJ):
	$(CC) $(CFLAGS) -c $<

.cpp.$(OBJ):
	$(CC) $(CFLAGS) -c $<

.cu.$(OBJ):
	$(CC) $(CFLAGS) -c $<

matmul.o:  $(HEADERS_DEPS)
	$(CC) $(CFLAGS) -c matmul.cu

vadd.o:  $(HEADERS_DEPS)
	$(CC) $(CFLAGS) -c vadd.cu

#test: $(EXES)
#	for i in $(EXES); do \
#            $(PRE)$$i; \
#        done
