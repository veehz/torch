import { Tensor } from '../tensor';

export function linspace(start: number, end: number, steps: number) {
  const data = [];
  const step = (end - start) / (steps - 1);
  for (let i = 0; i < steps - 1; i++) {
    data.push(start + i * step);
  }
  data.push(end);
  return new Tensor(data);
}

export function arange(start: number, end: number = undefined, step: number = 1) {
  const data = [];
  if (end === undefined) {
    end = start;
    start = 0;
  }
  if (step === 0) {
    throw new Error('step must be nonzero');
  }
  if (Math.sign(end - start) !== Math.sign(step)) {
    throw new Error('upper bound and lower bound inconsistent with step sign');
  }
  for (let i = start; i < end; i += step) {
    data.push(i);
  }
  return new Tensor(data);
}
