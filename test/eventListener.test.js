import { assert } from 'chai';
import * as torch from 'torch';

describe('Event Bus', () => {
    const a = new torch.Tensor([1, 2, 3], { requires_grad: true });
    const b = new torch.Tensor([4, 5, 6], { requires_grad: true });
    const ab = a.add(b);
    const sum = ab.sum();
    
    describe('tensor.beforeBackward', () => {
        it('should dispatch event before backward', () => {
            let eventDispatched = false;
            torch.eventBus.addEventListener(torch.events.TENSOR_BEFORE_BACKWARD, () => {
                eventDispatched = true;
            });
            sum.backward();
            assert.isTrue(eventDispatched);
        });
    });

    describe('tensor.afterBackward', () => {
        it('should dispatch event after backward', () => {
            let eventDispatched = false;
            torch.eventBus.addEventListener(torch.events.TENSOR_AFTER_BACKWARD, () => {
                eventDispatched = true;
            });
            sum.backward();
            assert.isTrue(eventDispatched);
        });
    });

    describe('operation.beforeForward', () => {
        it('should dispatch event before forward', () => {
            let eventDispatched = false;
            torch.eventBus.addEventListener(torch.events.OPERATION_BEFORE_FORWARD, () => {
                eventDispatched = true;
            });
            const c = a.add(b);
            assert.isTrue(eventDispatched);
        });
    });

    describe('operation.afterForward', () => {
        it('should dispatch event after forward', () => {
            let eventDispatched = false;
            torch.eventBus.addEventListener(torch.events.OPERATION_AFTER_FORWARD, () => {
                eventDispatched = true;
            });
            const c = a.add(b);
            assert.isTrue(eventDispatched);
        });
    });

    describe('operation.beforeBackward', () => {
        it('should dispatch event before backward', () => {
            let eventDispatched = false;
            torch.eventBus.addEventListener(torch.events.OPERATION_BEFORE_BACKWARD, () => {
                eventDispatched = true;
            });
            sum.backward();
            assert.isTrue(eventDispatched);
        });
    });

    describe('operation.afterBackward', () => {
        it('should dispatch event after backward', () => {
            let eventDispatched = false;
            torch.eventBus.addEventListener(torch.events.OPERATION_AFTER_BACKWARD, () => {
                eventDispatched = true;
            });
            sum.backward();
            assert.isTrue(eventDispatched);
        });
    });

    describe('operation.accumulateGrad', () => {
        it('should dispatch event after accumulateGrad', () => {
            let eventDispatched = false;
            torch.eventBus.addEventListener(torch.events.OPERATION_ACCUMULATE_GRAD, () => {
                eventDispatched = true;
            });
            sum.backward();
            assert.isTrue(eventDispatched);
        });
    });
});