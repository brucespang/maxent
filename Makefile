CC=clang
CFLAGS += -I/usr/include/python2.7 -I/usr/local/include/gsl -L/usr/local/lib -g -Wall -Wextra -Werror -fPIC -O3
LDFLAGS += -lpython2.7 -lgsl -lgslcblas -lphenom

all: cmaxent.so run

sources  := $(wildcard *.c)
includes := $(wildcard *.h)
objects  := $(sources:%.c=%.o)

$(objects): %.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@

cmaxent.so: $(objects)
	$(CC) -shared $(CFLAGS) $^ -o cmaxent.so $(LDFLAGS)

run: cmaxent.so
	LD_LIBRARY_PATH=/usr/local/lib time python maxent.py
	LD_LIBRARY_PATH=/usr/local/lib time python gender.py

clean:
	rm cmaxent.so
