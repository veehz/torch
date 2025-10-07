import { registerOperation } from './operations/registry';
import { Tensor } from './tensor';

import { Add, Div, Log, Mul, Pow, Sub, Sum } from './operations/basic';

import { _broadcast_shape, _get_original_index_gpu } from './broadcasting';

export { Tensor };

registerOperation('add', Add);
registerOperation('mul', Mul);
registerOperation('sum', Sum);
registerOperation('sub', Sub);
registerOperation('pow', Pow);
registerOperation('log', Log);
registerOperation('div', Div);