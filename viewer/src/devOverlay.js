// Dev overlay: fps, frame ms, connection, worker state, last message type and last validation error.
//
// fps and frame ms are averaged over the last refresh window (REFRESH_MS). frame ms is the mean
// interval between frames; cpu ms is the mean time spent inside the frame callback + render.
// The DOM is rewritten only once per window, so the overlay does not cost frames itself.

const REFRESH_MS = 500;

export function createDevOverlay(el) {
  let windowStart = null;
  let frames = 0;
  let cpuSum = 0;
  let conn = "connecting";
  let fps = 0;
  let frameMs = 0;
  let cpuMs = 0;

  function render(status) {
    el.textContent = [
      `fps        ${fps.toFixed(1)}`,
      `frame ms   ${frameMs.toFixed(2)}   cpu ms ${cpuMs.toFixed(2)}`,
      `ws         ${conn}`,
      `worker     ${status.worker ?? "-"}`,
      `messages   ${status.received} (${status.rejected} rejected)`,
      `last type  ${status.lastType ?? "-"}`,
      `last error ${status.lastError ?? "-"}`,
    ].join("\n");
  }

  return {
    setConnection(state) {
      conn = state;
    },
    /** Call once per frame with the frame timestamp and the CPU ms it took; status = handle.status. */
    frame(now, cpu, status) {
      if (windowStart === null) {
        windowStart = now;
        render(status);
        return;
      }
      frames += 1;
      cpuSum += cpu;
      const dt = now - windowStart;
      if (dt < REFRESH_MS) return;
      fps = (1000 * frames) / dt;
      frameMs = dt / frames;
      cpuMs = cpuSum / frames;
      windowStart = now;
      frames = 0;
      cpuSum = 0;
      render(status);
    },
  };
}
