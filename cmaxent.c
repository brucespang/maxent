#include "cmaxent.h"

#include <gsl_multimin.h>
#include <math.h>

#define NUM_ITERATIONS 100

typedef PyObject feature_t;

typedef struct {
  PyObject* features;
  uint32_t num_features;
  PyObject* classes;
  uint32_t num_classes;
  gsl_vector* weights;
} model_t;

typedef struct {
  PyObject* class;
  PyObject* data;
} observation_t;

struct params {
  model_t* model;
  observation_t* data;
  uint32_t num_data;
};

/* static void dump_vector(const gsl_vector* vec) { */
/*   printf("["); */
/*   for (uint32_t i = 0; i < vec->size; i++) { */
/*     printf("%f, ", gsl_vector_get(vec, i)); */
/*   } */
/*   printf("]\n"); */
/* } */

static double call_feature(feature_t* feature, const observation_t* obs) {
  PyObject* args = PyTuple_Pack(2, obs->class, obs->data);
  PyObject* result = PyEval_CallObject(feature, args);
  Py_DECREF(args);
  if (result == NULL) {
    fprintf(stderr, "error in feature\n");
    return 0;
  } else {
    double res = PyFloat_AsDouble(result);
    Py_DECREF(result);
    return res;
  }
}

static double feature_sum(model_t* model, const gsl_vector* weights, const observation_t* obs) {
  double sum = 0;
  for (uint32_t i = 0; i < model->num_features; i++) {
    double f = call_feature(PyList_GET_ITEM(model->features, i), obs);
    sum += gsl_vector_get(weights, i) * f;
  }
  return sum;
}

static double likelihood(model_t* model, const gsl_vector* weights, const observation_t* actual) {
  double actuals = exp(feature_sum(model, weights, actual));
  /* printf("actuals: %f\n", actuals); */

  observation_t obs = {
    .class = NULL,
    .data = actual->data
  };
  double expecteds = 0;
  for (uint32_t i = 0; i < model->num_classes; i++) {
    PyObject* class = PyList_GET_ITEM(model->classes, i);
    obs.class = class;
    expecteds += exp(feature_sum(model, weights, &obs));
  }
  /* printf("expecteds: %f\n", expecteds); */

  /* printf("likelihood: %f\n", actuals/expecteds); */
  return actuals/expecteds;
}

static double log_likelihood(model_t* model, const gsl_vector* weights,
                             const observation_t* data, uint32_t num_data) {
  double ll = 0;

  for (uint32_t i = 0; i < num_data; i++) {
    ll += log(likelihood(model, weights, &data[i]));
  }

  /* printf("ll: %f\n", ll); */

  return ll;
}

static double func(const gsl_vector* weights, void* params) {
  struct params* ps = params;
  double ll = -1*log_likelihood(ps->model, weights, ps->data, ps->num_data);
  return ll;
}

static double empirical_count(feature_t* feature, const observation_t* data, uint32_t num_data) {
  double count = 0;
  for (uint32_t i = 0; i < num_data; i++) {
    count += call_feature(feature, &data[i]);
  }
  /* printf("empirical: %f\n", count); */
  return count;
}

static double predicted_count(model_t* model, feature_t* feature,
                              const gsl_vector* weights,
                              const observation_t* data, uint32_t num_data) {
  double count = 0;

  for (uint32_t i = 0; i < num_data; i++) {
    observation_t obs = {
      .class = NULL,
      .data = data[i].data
    };

    for (uint32_t j = 0; j < model->num_classes; j++) {
      obs.class = PyList_GET_ITEM(model->classes, j);
      count += likelihood(model, weights, &obs)*call_feature(feature, &obs);
    }
  }

  /* printf("predicted: %f\n", count); */
  return count;
}

static void gradient(const gsl_vector* weights, void* params, gsl_vector* grad) {
  struct params* ps = params;
  model_t* model = ps->model;
  observation_t* data = ps->data;
  uint32_t num_data = ps->num_data;

  for (uint32_t i = 0; i < model->num_features; i++) {
    feature_t* feature = PyList_GET_ITEM(model->features, i);
    double g_i = empirical_count(feature, data, num_data) - predicted_count(model, feature, weights, data, num_data);
    /* printf("g_i: %f\n", g_i); */

    gsl_vector_set(grad, i, -1*g_i);
  }
  /* dump_vector(grad); */
}

static void fdf(const gsl_vector* weights, void* params, double* f, gsl_vector* grad) {
  *f = func(weights, params);
  gradient(weights, params, grad);
}

static PyObject* train(model_t* model, observation_t* data, uint32_t num_data) {
  struct params ps = {
    .model = model,
    .data = data,
    .num_data = num_data
  };

  gsl_multimin_function_fdf f = {&func, &gradient, &fdf, model->num_features, &ps};

  gsl_vector* x = gsl_vector_alloc(f.n);
  for (uint32_t i = 0; i < f.n; i++) {
    gsl_vector_set(x, i, 0);
  }

  const gsl_multimin_fdfminimizer_type* T = gsl_multimin_fdfminimizer_vector_bfgs2;
  gsl_multimin_fdfminimizer* s = gsl_multimin_fdfminimizer_alloc(T, f.n);

  gsl_multimin_fdfminimizer_set(s, &f, x, 0.01, 0.1);

  int32_t status;
  uint32_t iter = 0;
  do {
    iter++;
    status = gsl_multimin_fdfminimizer_iterate(s);
    /* printf("%i: \n",iter); */
    /* printf("x "); dump_vector(s->x); */
    /* printf("g "); dump_vector(s->gradient); */
    /* printf("f(x) %g\n",s->f); */
    /* printf("status=%d\n", status); */
    /* printf("\n"); */
    if (status == GSL_ENOPROG) {
      break;
    }

    status = gsl_multimin_test_gradient(s->gradient, 1e-3);
  } while (iter < NUM_ITERATIONS && status == GSL_CONTINUE);

  PyObject* list = PyList_New(f.n);
  for (uint32_t i = 0; i < f.n; i++) {
    PyList_SET_ITEM(list, i, PyFloat_FromDouble(gsl_vector_get(s->x, i)));
  }
  gsl_multimin_fdfminimizer_free(s);

  return list;
}

void extract_model(model_t* model, PyObject* py_model) {
  PyObject* features = PyObject_GetAttrString(py_model, "features");
  assert(features);

  model->num_features = PyList_Size(features);
  model->features = features;

  PyObject* classes = PyObject_GetAttrString(py_model, "classes");
  assert(classes);

  model->num_classes = PyList_Size(classes);
  model->classes = classes;

  PyObject* py_weights = PyObject_GetAttrString(py_model, "weights");
  assert(py_weights);
  assert(PyList_GET_SIZE(py_weights) == model->num_features);

  model->weights = gsl_vector_alloc(model->num_features);
  for (uint32_t i = 0; i < model->num_features; i++) {
    gsl_vector_set(model->weights, i, PyFloat_AsDouble(PyList_GET_ITEM(py_weights, i)));
  }
}

void free_model(model_t* model) {
  Py_DECREF(model->features);
  Py_DECREF(model->classes);
  gsl_vector_free(model->weights);
}

PyObject* py_likelihood(PyObject* self, PyObject* args) {
  (void)self;

  PyObject *py_model, *py_data;
  if (!PyArg_ParseTuple(args, "OO", &py_model, &py_data))
    return NULL;

  model_t model;
  extract_model(&model, py_model);

  observation_t obs = {
    .class = PyTuple_GetItem(py_data, 0),
    .data = PyTuple_GetItem(py_data, 1)
  };

  double l = likelihood(&model, model.weights, &obs);

  free_model(&model);

  return Py_BuildValue("d", l);
}

PyObject* py_train(PyObject* self, PyObject* args) {
  (void) self;

  PyObject *py_model, *py_data;
  if (!PyArg_ParseTuple(args, "OO", &py_model, &py_data))
    return NULL;

  model_t model;
  extract_model(&model, py_model);

  uint32_t num_data = PyList_Size(py_data);
  observation_t* data = calloc(num_data, sizeof(observation_t));
  for (uint32_t i = 0; i < num_data; i++) {
    PyObject* d = PyList_GET_ITEM(py_data, i);
    data[i].class = PyTuple_GetItem(d, 0);
    data[i].data = PyTuple_GetItem(d, 1);
  }
  PyObject* weights = train(&model, data, num_data);

  free_model(&model);

  return weights;
}
