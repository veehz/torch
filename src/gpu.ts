/*
 * TODO:
 * - Probably use Source Academy gpu.js
 */

import { GPU, Texture } from '@veehz/gpu.js';
import {
  _get_original_index_from_transposed_index,
  _get_original_index_kernel
} from './broadcasting';

export { Texture };

/**
 * Problems with gpu:
 * - source array too large when all the options dynamicoutput dynamicarguments pipeline immutable are true
 * - if pipeline is not used, then the memory transfer is too slow
 * - if dynamicarguments is not used, then the source array is too large error will appear (https://github.com/gpujs/gpu.js/issues/495)
 * - if dynamicoutputs is not used, then loops and broadcasting cannot be used
 * 
 * CPU is currently fast enough as well.
 */
const gpu = new GPU({ mode: 'cpu' });

// gpu.addFunction(_get_original_index_kernel);
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

// for debugging purposes
export { GPU };
