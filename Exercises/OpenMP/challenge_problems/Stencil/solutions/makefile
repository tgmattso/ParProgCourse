#
#  USAGE:
#     make          ... to build the program
#     make test     ... to run the default test case
#
include ../make.def

EXES= stencil_par$(EXE) 

STN_PAR_OBJS  = stencil_par.$(OBJ)  

all: $(EXES)

stencil_par$(EXE): $(STN_PAR_OBJS) 
	$(CLINKER) $(CFLAGS) -o stencil_par$(EXE) $(STN_PAR_OBJS) $(LIBS)

test: $(EXES)
	for i in $(EXES); do \
            $(PRE)$$i; \
        done

clean:
	$(RM) $(EXES) *.$(OBJ)

.SUFFIXES:
.SUFFIXES: .c .cpp  .$(OBJ)

.c.$(OBJ):
	$(CC) $(CFLAGS) -c $<

.cpp.$(OBJ):
	$(CC) $(CFLAGS) -c $<
