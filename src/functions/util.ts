export function _get_strides(shape: number[]): number[] {
  const strides = new Array(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

export function _unravel_index(index: number, strides: number[]): number[] {
  return strides.map((stride) => {
    const coord = Math.floor(index / stride);
    index %= stride;
    return coord;
  });
}

export function _ravel_index(coords: number[], strides: number[]): number {
  return coords.reduce((acc, coord, i) => acc + coord * strides[i], 0);
}

export function _get_reduction_shape(
  shape: number[],
  dim?: number | number[],
  keepdim: boolean = false
): number[] {
  if (dim === undefined) return keepdim ? shape.map(() => 1) : [];
  
  const dims = Array.isArray(dim) ? dim : [dim];
  const normalized_dims = dims.map((d) => (d < 0 ? d + shape.length : d));

  if (keepdim) {
    return shape.map((s, i) => (normalized_dims.includes(i) ? 1 : s));
  } else {
    return shape.filter((_, i) => !normalized_dims.includes(i));
  }
}