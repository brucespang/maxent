#include "cmaxent.h"

#include <gsl_multimin.h>
#include <math.h>

#define NUM_ITERATIONS 100

typedef struct {
  feature_t* features;
  uint64_t num_features;
  PyObject* classes;
  uint64_t num_classes;
  gsl_vector* weights;
} model_t;

struct params {
  model_t* model;
  observation_t* data;
  uint64_t num_data;
};

static cache_entry_t* hs_get(ck_hs_t* hs, const feature_t* feature, const observation_t* obs) {
  cache_entry_t entry;
  memcpy(&entry.key.obs, obs, sizeof(observation_t));
  memcpy(&entry.key.feature, feature, sizeof(feature_t));
  return ck_hs_get(hs, CK_HS_HASH(hs, hs_hash, &entry), &entry);
}

static bool hs_put(ck_hs_t* hs, cache_entry_t* entry) {
  return ck_hs_put(hs, CK_HS_HASH(hs, hs_hash, entry), entry);
}

static double call_feature(feature_t* feature, const observation_t* obs) {
  cache_entry_t* entry;
  if (!(entry = hs_get(&feature_cache, feature, obs))) {
    feature_calls++;

    PyObject* args = PyTuple_Pack(2, obs->class, obs->data);
    PyObject* result = PyEval_CallObject(feature->func, args);
    Py_DECREF(args);
    double res;
    if (result == NULL) {
      fprintf(stderr, "error in feature\n");
      res = 0;
    } else {
      res = PyFloat_AsDouble(result);
      Py_DECREF(result);
    }

    entry = calloc(1, sizeof(cache_entry_t));
    memcpy(&entry->key.obs, obs, sizeof(observation_t));
    memcpy(&entry->key.feature, feature, sizeof(feature_t));
    entry->data = res;

    Py_INCREF(entry->key.feature.func);
    Py_INCREF(entry->key.obs.class);
    Py_INCREF(entry->key.obs.data);

    hs_put(&feature_cache, entry);
  }

  return entry->data;
}

static double feature_sum(model_t* model, const gsl_vector* weights, const observation_t* obs) {
  double sum = 0;
  for (uint32_t i = 0; i < model->num_features; i++) {
    double f = call_feature(&model->features[i], obs);
    sum += gsl_vector_get(weights, i) * f;
  }
  return sum;
}

static double likelihood(model_t* model, const gsl_vector* weights,
                         const observation_t* actual) {
  double actuals = exp(feature_sum(model, weights, actual));

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

  return actuals/expecteds;
}

static double log_likelihood(model_t* model, const gsl_vector* weights,
                             const observation_t* data, uint64_t num_data) {
  double ll = 0;

  for (uint32_t i = 0; i < num_data; i++) {
    ll += log(likelihood(model, weights, &data[i]));
  }

  return ll;
}

static double func(const gsl_vector* weights, void* params) {
  struct params* ps = params;
  double ll = -1*log_likelihood(ps->model, weights, ps->data, ps->num_data);
  return ll;
}

static double empirical_count(feature_t* feature, const observation_t* data, uint64_t num_data) {
  double count = 0;
  for (uint32_t i = 0; i < num_data; i++) {
    count += call_feature(feature, &data[i]);
  }
  return count;
}

static double predicted_count(model_t* model, feature_t* feature,
                              const gsl_vector* weights,
                              const observation_t* data, uint64_t num_data) {
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

  return count;
}

static void gradient(const gsl_vector* weights, void* params, gsl_vector* grad) {
  struct params* ps = params;
  model_t* model = ps->model;
  observation_t* data = ps->data;
  uint64_t num_data = ps->num_data;

  for (uint32_t i = 0; i < model->num_features; i++) {
    feature_t* feature = &model->features[i];
    double g_i = empirical_count(feature, data, num_data) - predicted_count(model, feature, weights, data, num_data);

    gsl_vector_set(grad, i, -1*g_i);
  }
}

static void fdf(const gsl_vector* weights, void* params, double* f, gsl_vector* grad) {
  *f = func(weights, params);
  gradient(weights, params, grad);
}

static PyObject* train(model_t* model, observation_t* data, uint64_t num_data) {
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
    if (status == GSL_ENOPROG) {
      break;
    }
    status = gsl_multimin_test_gradient(s->gradient, 0.1);
  } while (iter < NUM_ITERATIONS && status == GSL_CONTINUE);

  PyObject* list = PyList_New((int64_t)f.n);
  for (uint32_t i = 0; i < f.n; i++) {
    PyList_SET_ITEM(list, i, PyFloat_FromDouble(gsl_vector_get(s->x, i)));
  }
  gsl_multimin_fdfminimizer_free(s);

  return list;
}

static void extract_model(model_t* model, PyObject* py_model) {
  PyObject* feature_funcs = PyObject_GetAttrString(py_model, "features");
  assert(feature_funcs);

  model->num_features = (uint64_t)PyList_Size(feature_funcs);
  model->features = calloc(model->num_features, sizeof(feature_t));
  for (uint32_t i = 0; i < model->num_features; i++) {
    model->features[i].func = PyList_GET_ITEM(feature_funcs, i);
  }

  PyObject* classes = PyObject_GetAttrString(py_model, "classes");
  assert(classes);

  model->num_classes = (uint64_t)PyList_Size(classes);
  model->classes = classes;

  PyObject* py_weights = PyObject_GetAttrString(py_model, "weights");
  assert(py_weights);
  assert((uint64_t)PyList_GET_SIZE(py_weights) == model->num_features);

  model->weights = gsl_vector_alloc(model->num_features);
  for (uint32_t i = 0; i < model->num_features; i++) {
    gsl_vector_set(model->weights, i, PyFloat_AsDouble(PyList_GET_ITEM(py_weights, i)));
  }
}

static void free_model(model_t* model) {
  Py_DECREF(model->features);
  Py_DECREF(model->classes);
  gsl_vector_free(model->weights);
}

PyObject* py_likelihood(PyObject* self, PyObject* args) {
  (void)self;

  PyObject *py_model, *py_data;
  if (!PyArg_ParseTuple(args, "OO", &py_model, &py_data)) {
    return NULL;
  }

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
  if (!PyArg_ParseTuple(args, "OO", &py_model, &py_data)) {
    return NULL;
  }

  model_t model;
  extract_model(&model, py_model);

  uint64_t num_data = (uint64_t)PyList_Size(py_data);
  observation_t* data = calloc(num_data, sizeof(observation_t));
  for (uint32_t i = 0; i < num_data; i++) {
    PyObject* d = PyList_GET_ITEM(py_data, i);
    data[i].class = PyTuple_GetItem(d, 0);
    data[i].data = PyTuple_GetItem(d, 1);
  }
  PyObject* weights = train(&model, data, num_data);

  printf("%lu\n", feature_calls);

  free_model(&model);

  return weights;
}
