#
#  USAGE:
#     make          ... to build the program
#     make test     ... to run the default test case
#     make clean    ... remove object and executable files.
#

# verify that you are including the right make.def file for the platform
include make.def

COMMON      = Cpp_common

EXES     = vadd$(EXE) matmul$(EXE)

MMULOBJS = matmul.$(OBJ) matrix_lib.$(OBJ)

COMM_OBJS = $(COMMON)/wtime.$(OBJ)

all: $(EXES)

vadd$(EXE): vadd.$(OBJ)  $(COMM_OBJS) vadd.cl
	$(CLINKER) $(CFLAGS) $(OPENCLFLAGS) -o vadd$(EXE) vadd.$(OBJ) \
                         $(COMM_OBJS) $(LIBS)

matmul$(EXE): $(MMULOBJS) $(COMM_OBJS) 
	$(CLINKER) $(CFLAGS) $(OPENCLFLAGS) -o matmul$(EXE) $(MMULOBJS) \
                         $(COMM_OBJS) $(LIBS)

$(COMMON)/wtime.$(OBJ): $(COMMON)/wtime.c
	$(CC) -c $(CFLAGS) -o $(COMMON)/wtime.$(OBJ)  $(COMMON)/wtime.c

$(COMMON)/err_code.$(OBJ): $(COMMON)/err_code.c
	$(CC) -c $(CFLAGS) -o $(COMMON)/err_code.$(OBJ)  $(COMMON)/err_code.c

test: $(EXES)
	$(PRE)vadd$(EXE); \
        $(PRE)matmul$(EXE)

clean:
	$(RM) $(EXES) *.$(OBJ) $(COMM_OBJS)

veryclean:
	$(RM) $(EXES) $(COMM_OBJS) *.$(OBJ)

.SUFFIXES:
.SUFFIXES: .c .cpp .$(OBJ)

.c.$(OBJ):
	$(CC) $(CFLAGS) -c $<

.cpp.$(OBJ):
	$(CC) $(CFLAGS) -c $<

#test: $(EXES)
#	for i in $(EXES); do \
#            $(PRE)$$i; \
#        done
