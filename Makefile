CC=clang
CFLAGS += -I/usr/include/python2.7 -I/usr/local/include/gsl -L/usr/local/lib -g -Wall -Wextra -Werror -fPIC -O3 -std=gnu99
LDFLAGS += -lpython2.7 -lgsl -lgslcblas

all: cmaxent.so run

cmaxent.so: cmaxent.c cmaxent.h
	$(CC) -shared $(CFLAGS) -o cmaxent.so cmaxent.c $(LDFLAGS)

run: cmaxent.so
	python new_maxent.py
	python gender.py

clean:
	rm cmaxent.so
