import { useEffect, useRef, useState } from 'react';

export type RunStream = {
  epochs: any[];
  status: any;
  flags: any[];
  live: boolean; // true when connected via WebSocket
};

/**
 * Live run data. Prefers a WebSocket push stream; falls back to 3s polling if
 * the socket cannot connect. Returns epochs, status, flags.
 */
export function useRunStream(runId: string): RunStream {
  const [epochs, setEpochs] = useState<any[]>([]);
  const [status, setStatus] = useState<any>({});
  const [flags, setFlags] = useState<any[]>([]);
  const [live, setLive] = useState(false);
  const pollRef = useRef<number | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    // Set once the server sends {type: "done"} and closes cleanly (run
    // completed - nothing will ever change again). Prevents onclose from
    // treating that as a dropped connection and falling back to polling
    // forever on a run that's already finished.
    let finished = false;

    const poll = () => {
      fetch(`/api/runs/${runId}/epochs`).then((r) => r.json()).then((d) => !cancelled && setEpochs(d)).catch(() => {});
      fetch(`/api/runs/${runId}`).then((r) => r.json()).then((d) => !cancelled && setStatus(d.status || {})).catch(() => {});
      fetch(`/api/runs/${runId}/flags`).then((r) => r.json()).then((d) => !cancelled && setFlags(d)).catch(() => {});
    };

    const startPolling = () => {
      if (finished || pollRef.current != null) return;
      poll();
      pollRef.current = window.setInterval(poll, 3000);
    };
    const stopPolling = () => {
      if (pollRef.current != null) { window.clearInterval(pollRef.current); pollRef.current = null; }
    };

    // Try WebSocket first
    try {
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const ws = new WebSocket(`${proto}://${window.location.host}/ws/runs/${runId}`);
      wsRef.current = ws;

      ws.onopen = () => { if (!cancelled) { setLive(true); stopPolling(); } };
      ws.onmessage = (ev) => {
        if (cancelled) return;
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'update') {
            if (Array.isArray(msg.epochs)) setEpochs(msg.epochs);
            if (msg.status) setStatus(msg.status);
            if (Array.isArray(msg.flags)) setFlags(msg.flags);
          } else if (msg.type === 'done') {
            finished = true;
            stopPolling();
          }
        } catch { /* ignore */ }
      };
      ws.onerror = () => { if (!cancelled && !finished) { setLive(false); startPolling(); } };
      ws.onclose = () => { if (!cancelled && !finished) { setLive(false); startPolling(); } };
    } catch {
      startPolling();
    }

    // Always poll once immediately so the page isn't empty before WS connects
    poll();

    return () => {
      cancelled = true;
      stopPolling();
      try { wsRef.current?.close(); } catch { /* ignore */ }
    };
  }, [runId]);

  return { epochs, status, flags, live };
}
