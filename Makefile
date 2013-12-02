CC=clang
CFLAGS += -I/usr/include/python2.7 -I/usr/local/include/gsl -g -Wall -Wextra -Werror -fPIC -O3
LDFLAGS += -lpython2.7 -lgsl -lgslcblas

all: cmaxent.so run

sources  := $(wildcard *.c)
includes := $(wildcard *.h)
objects  := $(sources:%.c=%.o)

$(objects): %.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@

cmaxent.so: $(objects)
	$(CC) -shared $(CFLAGS) $^ -o cmaxent.so $(LDFLAGS)

run: cmaxent.so
	time python maxent.py
	time python gender.py

clean:
	rm cmaxent.so
	rm cmaxent.o
