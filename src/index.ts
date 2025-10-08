import { registerOperation } from './operations/registry';
import { Tensor } from './tensor';

import { Add, Div, Log, Mul, Pow, Sub, Sum } from './operations/classes';

export * from './operations/functional';
export { Tensor };

registerOperation('add', Add);
registerOperation('sub', Sub);
registerOperation('mul', Mul);
registerOperation('div', Div);
registerOperation('sum', Sum);
registerOperation('pow', Pow);
registerOperation('log', Log);
