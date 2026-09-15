CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -fPIC
CUDA_INC ?= /usr/include

all: memserver libmemhook.so

memserver: memserver.c proto.h
	$(CC) $(CFLAGS) -o $@ memserver.c -lrt -lpthread

libmemhook.so: libmemhook.c proto.h
	$(CC) $(CFLAGS) -I$(CUDA_INC) -shared -o $@ libmemhook.c -ldl -lrt

clean:
	rm -f memserver libmemhook.so

.PHONY: all clean
