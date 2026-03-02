import { Tensor } from "../build/node/torch.node.es.js";

const x = new Tensor([2.0], { requires_grad: true });
const y = x.pow(new Tensor([2.0]));

console.log(y.item());
y.backward();

console.log(x.grad?.item());