// https://docs.pytorch.org/docs/stable/notes/broadcasting.html
export function _broadcast_shape(a_shape: number[], b_shape: number[]): number[] {
  const result_length = Math.max(a_shape.length, b_shape.length);
  const padded_a_shape = [...Array(result_length - a_shape.length).fill(1), ...a_shape];
  const padded_b_shape = [...Array(result_length - b_shape.length).fill(1), ...b_shape];

  const result_shape: number[] = [];

  for (let i = 0; i < result_length; i++) {
    if (
      padded_a_shape[i] !== padded_b_shape[i] &&
      padded_a_shape[i] !== 1 &&
      padded_b_shape[i] !== 1
    ) {
      throw new Error(`Shape mismatch: ${a_shape} and ${b_shape}`);
    }

    result_shape.push(Math.max(padded_a_shape[i], padded_b_shape[i]));
  }

  return result_shape;
}

export function _pad_shape(shape: number[], broadcast_shape: number[]): number[] {
  if(shape.length >= broadcast_shape.length) {
    return shape;
  }

  return [...Array(broadcast_shape.length - shape.length).fill(1), ...shape];
}

export function _get_original_index(
  original_shape: number[],
  new_shape: number[],
  index: number
): number {
  let original_index = 0;
  let cur_stride = 1;
  let temp_index = index;

  for (let i = original_shape.length - 1; i >= 0; i--) {
    if (original_shape[i] > 1) {
      const dim_index = temp_index % new_shape[i];
      original_index = original_index + dim_index * cur_stride;
    }
    cur_stride *= original_shape[i];
    temp_index = Math.floor(temp_index / new_shape[i]);
  }
  return original_index;
}

export function _get_original_index_kernel(
  original_shape: number[],
  new_shape: number[],
  index: number
): number {
  let original_index = 0;
  let cur_stride = 1;
  let temp_index = index;

  for (let i = this.constants.shape_length - 1; i >= 0; i--) {
    if (original_shape[i] > 1) {
      const dim_index = temp_index % new_shape[i];
      original_index = original_index + dim_index * cur_stride;
    }
    cur_stride = cur_stride * original_shape[i];
    temp_index = Math.floor(temp_index / new_shape[i]);
  }
  return original_index;
}

export function _get_original_index_from_transposed_index(
  original_shape: number[],
  dim0: number,
  dim1: number,
  transposed_index: number
): number {
  let original_index = 0;
  let cur_stride = 1;
  let temp_index = transposed_index;

  let dim0_index = 0;
  let dim1_index = 0;

  for (let i = this.constants.shape_length - 1; i >= 0; i--) {
    const dim_index = temp_index % original_shape[i];
    if(i == dim0) {
      dim0_index = dim_index;
    }
    if(i == dim1) {
      dim1_index = dim_index;
    }
    temp_index = Math.floor(temp_index / original_shape[i]);
  }

  temp_index = transposed_index;

  for (let j = this.constants.shape_length - 1; j >= 0; j--) {
    const dim_index = temp_index % original_shape[j];
    if(j == dim0) {
      original_index = original_index + dim1_index * cur_stride;
    } else if(j == dim1) {
      original_index = original_index + dim0_index * cur_stride;
    } else {
      original_index = original_index + dim_index * cur_stride;
    }
    cur_stride = cur_stride * original_shape[j];
    temp_index = Math.floor(temp_index / original_shape[j]);
  }

  return original_index;

  // return transposed_index;
}