#
# Makefile 
#-------------------------------------------------------------------------------
#
# Prerequisites:
# - update the path to eigen boost library EIGENDIR
#
# Usage: 
# - make so     : compile shared library only
# - make exe    : compile test executable only
# - make clean  : clean up 
#
# Compiler:
# - compiler, compiling and linking options can be changed via the CPPCOMP, 
#   LINKOPT and COMPOOPT.
# 
# Reference: 
# http://oreilly.com/catalog/make3/book/
# 
#-------------------------------------------------------------------------------



# ----------------------------------------------------------------- Source code

# Headers
HDIR=./include
# Source Folder
SRCDIR=./src


### objects
OBJDIR=./obj
OBJS=\
	$(OBJDIR)/lib_biot.o\
	$(OBJDIR)/cpp_interface.o\
	$(OBJDIR)/test_wrapper.o\



# External Libraries
# EIGENDIR=/home/sm6110/git/eigen-eigen-3.2.5
EIGENDIR=$(EIGEN3_INCLUDE_DIR)#eigen-3.3.4

# -------------------------------------------------------- Compiler and options 

# compiler
CPPCOMP= g++

# include headers and external libraries
#INCLOPT= -I"$(HDIR)" -I"$(EIGENDIR)" -I"$(BOOSTDIR)" 
INCLOPT= -I"$(HDIR)" -I"$(EIGENDIR)"

# Linking options
LINKOPT= -O3 -g3 -Wall -fopenmp -shared 

# Compiling options
# try also -Ofast...
COMPOPT= -O3 -g3 -Wall -fmessage-length=0 -fopenmp -fPIC -MMD -MP 



# ---------------------------------------------------------- Build dependencies
## Define make targets
#
# Automatic variables:
# $@: target filename
# $% filename element of an archive member specification
# $< filename of the first prerequisite
# $? names of all prerequisites that are newer than the target, separated by spaces. 

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@echo 'Building target: $@'	
	$(CPPCOMP) $(INCLOPT) $(COMPOPT) -c -o $@ $< 
	@echo 'Finished building target: $@'
	@echo ' '


# -------------------------------------------------------------------- Commands

# this needs to be below the definition of OBJS
exe: $(OBJS)
	@echo 'making exe...'
	### tester of python wrapper functions
	$(CPPCOMP) $(COMPTOPT) -I"$(EIGENDIR)" -o test_wrapper.exe $(OBJS)


so: $(OBJS)
	@echo 'making so...'
	$(CPPCOMP) $(LINKOPT)  -o cpplibs.so $(OBJS)


clean:
	rm -f ./obj/*.o
	rm -f ./obj/*.d	
	rm -f ./*.o
	rm -f ./*exe
	rm -f ./*so



















