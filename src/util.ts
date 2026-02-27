let globalId = 0;

export const getNextId = () => {
  return globalId++;
};

export const eventBus = new EventTarget();