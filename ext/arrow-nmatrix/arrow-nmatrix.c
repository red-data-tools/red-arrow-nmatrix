/*
 * Copyright 2017-2018 Kouhei Sutou <kou@clear-code.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <arrow-glib/arrow-glib.h>

#include <rbgobject.h>

#include <nmatrix.h>

/* TODO: NMatrix should extern them. */
extern const size_t DTYPE_SIZES[NM_NUM_DTYPES];
/* extern size_t nm_storage_count_max_elements(const STORAGE* storage); */

void Init_arrow_nmatrix(void);

static nm_dtype_t
garrow_type_to_nmatrix_dtype(GArrowType arrow_type)
{
  nm_dtype_t nmatrix_type = (nm_dtype_t)-1;

  switch (arrow_type) {
  case GARROW_TYPE_UINT8:
    nmatrix_type = BYTE;
    break;
  case GARROW_TYPE_INT8:
    nmatrix_type = INT8;
    break;
  case GARROW_TYPE_INT16:
    nmatrix_type = INT16;
    break;
  case GARROW_TYPE_INT32:
    nmatrix_type = INT32;
    break;
  case GARROW_TYPE_INT64:
    nmatrix_type = INT64;
    break;
  case GARROW_TYPE_FLOAT:
    nmatrix_type = FLOAT32;
    break;
  case GARROW_TYPE_DOUBLE:
    nmatrix_type = FLOAT64;
    break;
  case GARROW_TYPE_NA:
  case GARROW_TYPE_BOOLEAN:
  case GARROW_TYPE_UINT16:
  case GARROW_TYPE_UINT32:
  case GARROW_TYPE_UINT64:
  case GARROW_TYPE_HALF_FLOAT:
  case GARROW_TYPE_STRING:
  case GARROW_TYPE_BINARY:
  case GARROW_TYPE_FIXED_SIZE_BINARY:
  case GARROW_TYPE_DATE32:
  case GARROW_TYPE_DATE64:
  case GARROW_TYPE_TIMESTAMP:
  case GARROW_TYPE_TIME32:
  case GARROW_TYPE_TIME64:
  case GARROW_TYPE_INTERVAL_MONTHS:
  case GARROW_TYPE_INTERVAL_DAY_TIME:
  case GARROW_TYPE_DECIMAL128:
  case GARROW_TYPE_DECIMAL256:
  case GARROW_TYPE_LIST:
  case GARROW_TYPE_STRUCT:
  case GARROW_TYPE_SPARSE_UNION:
  case GARROW_TYPE_DENSE_UNION:
  case GARROW_TYPE_DICTIONARY:
  case GARROW_TYPE_MAP:
  case GARROW_TYPE_EXTENSION:
  case GARROW_TYPE_FIXED_SIZE_LIST:
  case GARROW_TYPE_DURATION:
  case GARROW_TYPE_LARGE_STRING:
  case GARROW_TYPE_LARGE_BINARY:
  case GARROW_TYPE_LARGE_LIST:
  default:
    break;
  }

  return nmatrix_type;
}

static VALUE
rb_arrow_tensor_to_nmatrix(VALUE self)
{
  GArrowTensor *tensor;
  GArrowType value_type;
  nm_dtype_t nmatrix_data_type;
  gint64 *shape;
  gint n_dimensions;
  GArrowBuffer *buffer;
  GBytes *data;
  gconstpointer data_raw;
  gsize data_size;
  VALUE rb_nmatrix = Qnil;

  tensor = RVAL2GOBJ(self);
  value_type = garrow_tensor_get_value_type(tensor);
  nmatrix_data_type = garrow_type_to_nmatrix_dtype(value_type);
  if (nmatrix_data_type == (nm_dtype_t)-1) {
    GArrowDataType *data_type;
    VALUE rb_data_type;
    data_type = garrow_tensor_get_value_data_type(tensor);
    rb_data_type = GOBJ2RVAL(data_type);
    g_object_unref(data_type);
    rb_raise(rb_eArgError,
             "Arrow::Tensor data type must be uint8, int*, float or double: "
             "<%" PRIsVALUE ">",
             rb_data_type);
  }

  shape = garrow_tensor_get_shape(tensor, &n_dimensions);
  buffer = garrow_tensor_get_buffer(tensor);
  data = garrow_buffer_get_data(buffer);
  data_raw = g_bytes_get_data(data, &data_size);

  rb_nmatrix = rb_nmatrix_dense_create(nmatrix_data_type,
                                       (size_t *)shape,
                                       n_dimensions,
                                       (void *)data_raw,
                                       data_size);
  g_bytes_unref(data);
  g_object_unref(buffer);
  g_free(shape);

  return rb_nmatrix;
}

static GArrowDataType *
nmatrix_dtype_to_garrow_data_type(nm_dtype_t nmatrix_type)
{
  GArrowDataType *arrow_data_type = NULL;

  switch (nmatrix_type) {
  case BYTE:
    arrow_data_type = GARROW_DATA_TYPE(garrow_uint8_data_type_new());
    break;
  case INT8:
    arrow_data_type = GARROW_DATA_TYPE(garrow_int8_data_type_new());
    break;
  case INT16:
    arrow_data_type = GARROW_DATA_TYPE(garrow_int16_data_type_new());
    break;
  case INT32:
    arrow_data_type = GARROW_DATA_TYPE(garrow_int32_data_type_new());
    break;
  case INT64:
    arrow_data_type = GARROW_DATA_TYPE(garrow_int64_data_type_new());
    break;
  case FLOAT32:
    arrow_data_type = GARROW_DATA_TYPE(garrow_float_data_type_new());
    break;
  case FLOAT64:
    arrow_data_type = GARROW_DATA_TYPE(garrow_double_data_type_new());
    break;
  case COMPLEX64:
  case COMPLEX128:
  case RUBYOBJ:
  default:
    break;
  }

  return arrow_data_type;
}

static VALUE
rb_nmatrix_to_arrow(VALUE self)
{
  GArrowDataType *data_type;
  GArrowBuffer *data;
  GArrowTensor *tensor;
  VALUE rb_tensor;

  data_type = nmatrix_dtype_to_garrow_data_type(NM_DTYPE(self));
  if (!data_type) {
    rb_raise(rb_eArgError,
             "NMatrix data type must be "
             ":byte, :int8, :int16, :int32, :int64, :float32 or :float64: "
             "<%" PRIsVALUE ">",
             self);
  }
  data = garrow_buffer_new((const guint8 *)NM_DENSE_ELEMENTS(self),
                           NM_SIZEOF_DTYPE(self) * NM_DENSE_COUNT(self));
  tensor = garrow_tensor_new(data_type,
                             data,
                             (gint64 *)(NM_STORAGE(self)->shape),
                             NM_DIM(self),
                             NULL,
                             0,
                             NULL,
                             0);
  g_object_unref(data);
  g_object_unref(data_type);

  rb_tensor = GOBJ2RVAL(tensor);
  g_object_unref(tensor);

  return rb_tensor;
}

void
Init_arrow_nmatrix(void)
{
  VALUE rb_Arrow;
  VALUE rb_ArrowTensor;

  rb_Arrow = rb_const_get(rb_cObject, rb_intern("Arrow"));
  rb_ArrowTensor = rb_const_get(rb_Arrow, rb_intern("Tensor"));

  rb_define_method(rb_ArrowTensor, "to_nmatrix",
                   rb_arrow_tensor_to_nmatrix, 0);

  rb_define_method(cNMatrix, "to_arrow",
                   rb_nmatrix_to_arrow, 0);
}
