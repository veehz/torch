/*
 * TODO:
 * - Probably use Source Academy gpu.js
 */

import { GPU } from '@veehz/gpu.js';
import { _get_original_index_kernel } from './broadcasting';

const gpu = new GPU();

gpu.addFunction(_get_original_index_kernel, {
    returnType: 'Integer',
    argumentTypes: {
        original_shape: 'Array',
        new_shape: 'Array',
        // shape_length: 'Integer',
        index: 'Integer'
    }
});

export default gpu;
