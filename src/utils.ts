import { Tensor } from './tensor';

export function ones_like(other: Tensor): Tensor {
    const tensor = new Tensor(Array(other.data.length).fill(1));
    tensor.shape = other.shape;
    return tensor;
}