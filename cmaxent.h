#ifndef CMAXENT_H_
#define CMAXENT_H_

#include <Python.h>
#include <ck_hs.h>
#include "murmurhash.h"

PyObject* py_likelihood(PyObject* self, PyObject* args);
PyObject* py_train(PyObject* self, PyObject* args);

ck_hs_t feature_cache;

static PyMethodDef cmaxent_methods[] = {
  {"likelihood", py_likelihood, METH_VARARGS, ""},
  {"train", py_train, METH_VARARGS, ""},
  {NULL, NULL, 0, NULL}
};

uint64_t feature_calls;

typedef struct {
  PyObject* class;
  PyObject* data;
} observation_t;

typedef struct {
  PyObject* func;
} feature_t;

typedef struct {
  observation_t obs;
  feature_t feature;
} cache_entry_key_t;

typedef struct {
  cache_entry_key_t key;
  double data;
} cache_entry_t;

static void * hs_malloc(size_t r) {
  return malloc(r);
}

static void hs_free(void *p, size_t b, bool r) {
  (void)b;
  (void)r;

  free(p);
}

struct ck_malloc hs_ck_malloc = {
  .malloc = hs_malloc,
  .free = hs_free
};

static bool hs_compare(const void* x, const void* y) {
  const cache_entry_t* a = x;
  const cache_entry_t* b = y;
  return memcmp(&a->key, &b->key, sizeof(cache_entry_key_t)) == 0;
}

static unsigned long hs_hash(const void* e, unsigned long seed) {
  const cache_entry_t* entry = e;
  uint64_t h[2];
  MurmurHash3_x64_128(&entry->key, sizeof(cache_entry_key_t), (uint32_t)seed, h);
  return h[0];
}

PyMODINIT_FUNC initcmaxent() {
  feature_calls = 0;
  ck_hs_init(&feature_cache, CK_HS_MODE_DIRECT | CK_HS_MODE_SPMC,
               hs_hash, hs_compare, &hs_ck_malloc, 4, 0);
  Py_InitModule("cmaxent", cmaxent_methods);
}

#endif  // CMAXENT_H_
