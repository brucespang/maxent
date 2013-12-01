#include "cmaxent.h"

#include <gsl_multimin.h>
#include <gsl_blas.h>
#include <math.h>

#define NUM_ITERATIONS 100
#define DEBUG

typedef struct {
  PyObject** features;
  uint64_t num_features;
  PyObject** classes;
  uint64_t num_classes;
  gsl_matrix* weights;
} model_t;

struct params {
  model_t* model;
  gsl_matrix* data;
  gsl_vector* labels;
  gsl_matrix* G_empirical;
  bool calc_func_value;
  bool calc_grad_value;
};

#ifdef DEBUG
static void gsl_matrix_dump(const gsl_matrix* M) {
  printf("[");
  for (uint32_t i = 0; i < M->size1; i++) {
    for (uint32_t j = 0; j < M->size2; j++) {
      if (!(i == 0 && j == 0)) {
        printf(" ");
      }
      printf("%f", gsl_matrix_get(M, i, j));
    }
    if (i != M->size1-1) {
      printf("\n");
    }
  }
  printf("]\n");
}
#endif

void gsl_vector_dump(const gsl_vector* x) {
  printf("[");
  for (uint32_t i = 0; i < x->size; i++) {
    printf("%f", gsl_vector_get(x, i));
    if (i != x->size-1) {
      printf(", ");
    }
  }
  printf("]");
}

static double call_feature(PyObject* feature, PyObject* data) {
  assert(PyCallable_Check(feature));

  PyObject* args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, data);
  PyObject* result = PyEval_CallObject(feature, args);
  if (result == NULL) {
    fprintf(stderr, "error in feature:\n");
    PyErr_Print();
    exit(1);
  } else {
    double res = PyFloat_AsDouble(result);
    Py_DECREF(result);
    return res;
  }
}

static inline size_t gsl_vector_size(const gsl_vector* x) {
  return x->size;
}

static inline size_t gsl_matrix_num_rows(const gsl_matrix* m) {
  return m->size1;
}

static inline size_t gsl_matrix_num_cols(const gsl_matrix* m) {
  return m->size2;
}

static inline gsl_matrix* gsl_matrix_mul(gsl_matrix* A, gsl_matrix* B) {
  gsl_matrix* C = gsl_matrix_alloc(A->size1, B->size2);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                 1, A, B,
                 0, C);
  return C;
}

// TODO: vectorize
static void gsl_matrix_map(gsl_matrix* X, double (*f)(double)) {
  for (uint32_t i = 0; i < gsl_matrix_num_cols(X); i++) {
    for (uint32_t j = 0; j < gsl_matrix_num_rows(X); j++) {
      gsl_matrix_set(X, j, i, f(gsl_matrix_get(X, j, i)));
    }
  }
}

static void gsl_vector_map(gsl_vector* x, double (*f)(double)) {
  for (uint32_t i = 0; i < gsl_vector_size(x); i++) {
    gsl_vector_set(x, i, f(gsl_vector_get(x, i)));
  }
}

// TODO: vectorize
gsl_vector* gsl_matrix_sum(gsl_matrix* X) {
  gsl_vector* res = gsl_vector_alloc(gsl_matrix_num_cols(X));
  for (uint32_t i = 0; i < gsl_matrix_num_cols(X); i++) {
    uint32_t sum = 0;
    for (uint32_t j = 0; j < gsl_matrix_num_rows(X); j++) {
      sum += gsl_matrix_get(X, j, i);
    }
    gsl_vector_set(res, i, sum);
  }
  return res;
}

static bool gsl_matrix_equals(const gsl_matrix* A, const gsl_matrix* B) {
  return A->block->size == B->block->size && memcmp(A->block->data, B->block->data,
                                                    A->block->size) == 0;
}

// normalize all columns in matrix by log(sum(exp(values)))
static void log_normalize(gsl_matrix* matrix) {
  // allocate a copy of matrix so that we can apply exp
  // without potentially fucking everything up.
  gsl_matrix* tmp = gsl_matrix_alloc(matrix->size1, matrix->size2);
  gsl_matrix_memcpy(tmp, matrix);
  assert(gsl_matrix_equals(tmp, matrix));

  // apply exp
  gsl_matrix_map(tmp, exp);
  // sum the values in each matrix column
  gsl_vector* logZ = gsl_matrix_sum(tmp);
  // take the log of each matrix column
  gsl_vector_map(logZ, log);

  // subtract the normalizing constant from each item in the matrix
  for (uint32_t i = 0; i < gsl_matrix_num_rows(matrix); i++) {
    gsl_vector r = gsl_matrix_row(matrix, i).vector;
    gsl_vector_sub(&r, logZ);
  }

  gsl_vector_free(logZ);
  gsl_matrix_free(tmp);
}

static void fdf(const gsl_vector* weights_vector, void* params,
                double* f, gsl_vector* gradient_vector) {
  struct params* ps = params;
  assert(ps->calc_func_value || ps->calc_grad_value);

  gsl_vector* weights_copy = gsl_vector_alloc(gsl_vector_size(weights_vector));
  gsl_vector_memcpy(weights_copy, weights_vector);
  assert(gsl_vector_equal(weights_vector, weights_copy));

  gsl_matrix weights = gsl_matrix_view_vector(weights_copy,
                                        ps->model->num_classes,
                                        ps->model->num_features).matrix;

  gsl_matrix* data = ps->data;

#ifdef DEBUG
  printf("\nfdf\n");

  printf("weights:\n");
  gsl_matrix_dump(&weights);
  printf("data:\n");
  gsl_matrix_dump(data);
#endif

  // matrix of scores for input data: rows are classes, columns are data,
  // value is f(d)
  gsl_matrix* scores = gsl_matrix_mul(&weights, data);

#ifdef DEBUG
  printf("scores:\n");
  gsl_matrix_dump(scores);
#endif

  // scores = Normalized log probability (LxN)
  log_normalize(scores);

#ifdef DEBUG
  printf("normalized:\n");
  gsl_matrix_dump(scores);
#endif

  // straight up horrible hack because GSL won't let us not define a func/gradient
  if (ps->calc_func_value) {
    // Function value: sum of log probabilities
    // TODO: vectorize
    double res = 0;
    for (uint32_t i = 0; i < gsl_matrix_num_cols(scores); i++) {
      res += gsl_matrix_get(scores, gsl_vector_get(ps->labels, i), i);
    }

    // we're using a minimization library.
    res *= -1;
    *f = res;

#ifdef DEBUG
    printf("f(x) = %f\n", *f);
#endif
  }

  if (ps->calc_grad_value) {
    gsl_matrix G = gsl_matrix_view_vector(gradient_vector,
                                          ps->model->num_classes,
                                          ps->model->num_features).matrix;

    // Probabilities (LxN)
    gsl_matrix_map(scores, exp);

    // Gradient (LxD)
    gsl_matrix* tmp = gsl_matrix_alloc(scores->size2, scores->size1);
    gsl_matrix_transpose_memcpy(tmp, scores);
    gsl_matrix* grad = gsl_matrix_mul(data, tmp);
    // G' = G_empirical - G, but in order to save memory
    // we can compute -1*(G - G_empirical) instead.
    // because we're using a minimization library, we'd then multiply by -1:
    // -1*-1*(G - G_empirical) = G - G_empirical.
    gsl_matrix_transpose_memcpy(&G, grad);

#ifdef DEBUG
    printf("empirical count:\n");
    gsl_matrix_dump(ps->G_empirical);
    printf("predicted count:\n");
    gsl_matrix_dump(&G);
#endif


    gsl_matrix_sub(&G, ps->G_empirical);

#ifdef DEBUG
    printf("gradient:\n");
    gsl_matrix_dump(&G);
#endif

    gsl_matrix_free(grad);
    gsl_matrix_free(tmp);
  }

  gsl_vector_free(weights_copy);
#ifdef DEBUG
  printf("\n");
#endif
}

static double func(const gsl_vector* weights, void* params) {
  struct params ps;
  memcpy(&ps, params, sizeof(struct params));
  ps.calc_grad_value = false;

  double res;
  fdf(weights, &ps, &res, NULL);

  return res;
}

static void grad(const gsl_vector* weights, void* params, gsl_vector* g) {
  struct params ps;
  memcpy(&ps, params, sizeof(struct params));
  ps.calc_func_value = false;

  fdf(weights, &ps, NULL, g);
}

static gsl_matrix* train(model_t* model, gsl_matrix* data, gsl_vector* labels) {
  gsl_matrix* empirical_count = gsl_matrix_alloc(model->num_classes,
                                                 model->num_features);
  assert(empirical_count->size1 == model->weights->size1 &&
         empirical_count->size2 == model->weights->size2);

  // TODO: vectorize
  for (uint32_t i = 0; i < model->num_classes; i++) {
    for (uint32_t k = 0; k < model->num_features; k++) {
      double sum = 0;
      for (uint32_t j = 0; j < labels->size; j++) {
        if (gsl_vector_get(labels, j) == i) {
          sum += gsl_matrix_get(data, k, j);
        }
      }
      gsl_matrix_set(empirical_count, i, k, sum);
    }
  }

  struct params ps = {
    .model = model,
    .data = data,
    .G_empirical = empirical_count,
    .labels = labels,
    .calc_func_value = true,
    .calc_grad_value = true
  };

  // we use gsl's multimin for matrix optimization by representing the gradient and
  // weight matrices as vectors by taking the underlying block of memory for the
  // matrix and viewing it as a vector.

  gsl_vector x = gsl_vector_view_array(model->weights->block->data,
                                       model->weights->block->size).vector;

  gsl_multimin_function_fdf f = {func, grad, &fdf, x.size, &ps};

  const gsl_multimin_fdfminimizer_type* T = gsl_multimin_fdfminimizer_vector_bfgs2;
  gsl_multimin_fdfminimizer* s = gsl_multimin_fdfminimizer_alloc(T, f.n);

  gsl_multimin_fdfminimizer_set(s, &f, &x, 0.01, 0.1);

  int32_t status;
  uint32_t iter = 0;
  do {
    iter++;
    status = gsl_multimin_fdfminimizer_iterate(s);

#ifdef DEBUG
    printf("%i: \n",iter);
    printf("x "); gsl_vector_dump(s->x); printf("\n");
    printf("g "); gsl_vector_dump(s->gradient); printf("\n");
    printf("f(x) %g\n",s->f);
    printf("status=%d\n", status);
    printf("\n");
#endif

    if (status == GSL_ENOPROG) {
      break;
    }
    status = gsl_multimin_test_gradient(s->gradient, 0.1);
  } while (iter < NUM_ITERATIONS && status == GSL_CONTINUE);

  gsl_matrix* weights = gsl_matrix_alloc(model->num_classes, model->num_features);
  memcpy(weights->block->data, s->x->block->data, s->x->size*sizeof(double));

  gsl_multimin_fdfminimizer_free(s);
  gsl_matrix_free(empirical_count);

  return weights;
}

static void extract_model(model_t* model, PyObject* py_model) {
  PyObject* feature_funcs = PyObject_GetAttrString(py_model, "features");
  assert(feature_funcs);

  model->num_features = (uint64_t)PyList_Size(feature_funcs);
  model->features = calloc(model->num_features, sizeof(PyObject*));
  for (uint32_t i = 0; i < model->num_features; i++) {
    model->features[i] = PyList_GET_ITEM(feature_funcs, i);
    Py_INCREF(model->features[i]);
  }
  Py_DECREF(feature_funcs);

  PyObject* classes = PyObject_GetAttrString(py_model, "classes");
  assert(classes);

  model->num_classes = (uint64_t)PyList_Size(classes);
  model->classes = calloc(model->num_classes, sizeof(PyObject*));
  for (uint32_t i = 0; i < model->num_classes; i++) {
    model->classes[i] = PyList_GET_ITEM(classes, i);
    Py_INCREF(model->classes[i]);
  }
  Py_DECREF(classes);

  PyObject* py_weights = PyObject_GetAttrString(py_model, "weights");
  assert(py_weights);
  assert((uint64_t)PyList_GET_SIZE(py_weights) == model->num_classes);

  model->weights = gsl_matrix_alloc(model->num_classes, model->num_features);
  for (uint32_t j = 0; j < model->num_classes; j++) {
    for (uint32_t i = 0; i < model->num_features; i++) {
      double w = PyFloat_AsDouble(PyList_GET_ITEM(PyList_GET_ITEM(py_weights, j), i));
      gsl_matrix_set(model->weights, j, i, w);
    }
  }
}

static void free_model(model_t* model) {
  for (uint32_t i = 0; i < model->num_features; i++) {
    Py_DECREF(model->features[i]);
  }
  for (uint32_t i = 0; i < model->num_classes; i++) {
    Py_DECREF(model->classes[i]);
  }
  gsl_matrix_free(model->weights);
}

PyObject* py_likelihood(PyObject* self, PyObject* args) {
  (void)self;

  PyObject *py_model, *py_obs;
  if (!PyArg_ParseTuple(args, "OO", &py_model, &py_obs)) {
    return NULL;
  }

  model_t model;
  extract_model(&model, py_model);

  gsl_matrix* X = gsl_matrix_alloc(model.num_features, 1);
  for (uint32_t i = 0; i < model.num_features; i++) {
    gsl_matrix_set(X, i, 0, call_feature(model.features[i], py_obs));
  }

  gsl_matrix* scores = gsl_matrix_mul(model.weights, X);
  /* gsl_matrix_map(scores, exp); */

  PyObject* res = PyList_New(scores->size1);
  for (uint32_t i = 0; i < scores->size1; i++) {
    PyList_SET_ITEM(res, i, PyFloat_FromDouble(gsl_matrix_get(scores, i, 0)));
  }

  free_model(&model);
  gsl_matrix_free(scores);

  return res;
}

int32_t get_class_id(model_t* model, PyObject* class) {
  for (uint32_t i = 0; i < model->num_classes; i++) {
    if (class == model->classes[i]) {
      return i;
    }
  }
  return -1;
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

  gsl_matrix* observations = gsl_matrix_alloc(model.num_features, num_data);
  gsl_vector* labels = gsl_vector_alloc(num_data);

  for (uint32_t i = 0; i < num_data; i++) {
    PyObject* item = PyList_GET_ITEM(py_data, i);
    PyObject* class = PyTuple_GetItem(item, 0);
    PyObject* obs  = PyTuple_GetItem(item, 1);

    gsl_vector_set(labels, i, get_class_id(&model, class));
    assert(gsl_vector_get(labels, i) != -1);

    for (uint32_t j = 0; j < model.num_features; j++) {
      gsl_matrix_set(observations, j, i, call_feature(model.features[j], obs));
    }
  }

  gsl_matrix* weights = train(&model, observations, labels);

  PyObject* rows = PyList_New(model.num_classes);
  for (uint32_t i = 0; i < model.num_classes; i++) {
    PyObject* row = PyList_New(model.num_features);
    for (uint32_t j = 0; j < model.num_features; j++) {
      PyList_SET_ITEM(row, j, PyFloat_FromDouble(gsl_matrix_get(weights, i, j)));
    }
    PyList_SET_ITEM(rows, i, row);
  }

  free_model(&model);
  gsl_vector_free(labels);

  return rows;
}
