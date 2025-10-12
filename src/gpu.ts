/*
 * TODO:
 * - Probably use Source Academy gpu.js
 */

import { GPU } from '@veehz/gpu.js';
import {
  _get_original_index_from_transposed_index,
  _get_original_index_kernel
} from './broadcasting';

const gpu = new GPU();

gpu.addFunction(_get_original_index_kernel, {
  returnType: 'Integer',
  argumentTypes: {
    original_shape: 'Array',
    new_shape: 'Array',
    index: 'Integer'
  }
});

gpu.addFunction(_get_original_index_from_transposed_index, {
  returnType: 'Integer',
  argumentTypes: {
    original_shape: 'Array',
    dim0: 'Integer',
    dim1: 'Integer',
    transposed_index: 'Integer'
  }
});

export default gpu;
