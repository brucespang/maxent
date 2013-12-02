#ifndef CMAXENT_H_
#define CMAXENT_H_

#include <Python.h>
#include <stdint.h>
#include <stdbool.h>

PyObject* py_likelihood(PyObject* self, PyObject* args);
PyObject* py_train(PyObject* self, PyObject* args);

static PyMethodDef cmaxent_methods[] = {
  {"likelihood", py_likelihood, METH_VARARGS, ""},
  {"train", py_train, METH_VARARGS, ""},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initcmaxent() {
  Py_InitModule("cmaxent", cmaxent_methods);
}

#endif  // CMAXENT_H_
