// Message worker: parse, validate and decode off the main thread (see src/net.js).
// Posts {ready: true} once its imports have loaded, then processMessage results in arrival order;
// field bits are transferred, not copied.

import { processMessage } from "./net.js";

self.onmessage = (ev) => {
  const result = processMessage(ev.data);
  if (result.ok && result.field) self.postMessage(result, [result.field.bits.buffer]);
  else self.postMessage(result);
};
self.postMessage({ ready: true });
