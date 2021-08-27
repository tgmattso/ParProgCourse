#
#  USAGE:
#     make          ... to build the program
#     make test     ... to run the default test case
#

# verify that you are including the right make.def file for the platform
include make.def

EXES     = jac_solv$(EXE) vadd$(EXE)

JAC_OBJS  = jac_solv.$(OBJ) mm_utils.$(OBJ)

VADD_OCL_OBJS  = vadd.$(OBJ)

all: $(EXES)

jac_solv$(EXE): $(JAC_OBJS) mm_utils.h
	$(CLINKER) $(CFLAGS) -o jac_solv$(EXE) $(JAC_OBJS) $(LIBS)

vadd$(EXE): $(VADD_OCL_OBJS)
	$(CLINKER) $(CFLAGS) -o vadd$(EXE) $(VADD_OCL_OBJS) $(LIBS) $(OCL_LIBS)

test: $(EXES)
	for i in $(EXES); do \
            $(PRE)$$i --device 1; \
        done

clean:
	$(RM) $(EXES) *.$(OBJ)

mm_testbed.$(OBJ): mm_utils.h
mm_utils.$(OBJ): mm_utils.h
mm_ijk.$(OBJ): mm_utils.h
mm_tst_cases.$(OBJ): mm_utils.h
gauss_solv.$(OBJ): mm_utils.h
jac_solv.$(OBJ): mm_utils.h
jac_solv_ocl_basic.$(OBJ): mm_utils.h
jac_solv_ocl_colmaj.$(OBJ): mm_utils.h
jac_solv_ocl_colmaj_nobr.$(OBJ): mm_utils.h
jac_solv_ocl_colmaj_nobr_wg.$(OBJ): mm_utils.h
jac_solv_ocl_best.$(OBJ): mm_utils.h


.SUFFIXES:
.SUFFIXES: .c .cpp .$(OBJ)

.c.$(OBJ):
	$(CC) $(CFLAGS) -c $<

.cpp.$(OBJ):
	$(CC) $(CFLAGS) -c $<


