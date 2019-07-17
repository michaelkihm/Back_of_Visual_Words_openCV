CC = g++ 
CFLAGS = -g -W
LDFLAGS = `pkg-config --libs opencv` `pkg-config --cflags opencv` 
BOOSTFLAGS = -L/usr/lib/x86_64-linux-gnu/ -lboost_system -lboost_filesystem 
CPPFLAGS = -I incl
BUILDDIR = build
VPATH = src 

#OBJ = main.o bar.o#function.o bar.o
OBJ = $(addprefix $(BUILDDIR)/, main.o datasethandler.o backofvisualwords.o)

vpath %.cpp src

prog: $(OBJ)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o prog $(OBJ) $(LDFLAGS) $(BOOSTFLAGS)



#%.o: %.cpp
#	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $(BUILDDIR)$@
$(BUILDDIR)/%.o: %.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@




.PHONY: clean
clean: 
	rm build/*.o; rm prog; echo finished clean
