import type { ExecutionEvent, PipelineDefinition, ValidationResult } from './types';

const DEFAULT_BASE = '/api';

function baseUrl(): string {
  if (typeof window === 'undefined') return DEFAULT_BASE;
  const override = (window as any).__ATOMIC_RUNNER_URL__ as string | undefined;
  return override ?? DEFAULT_BASE;
}

export async function validatePipeline(
  pipeline: PipelineDefinition
): Promise<ValidationResult> {
  const res = await fetch(`${baseUrl()}/validate`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(pipeline)
  });
  if (!res.ok) {
    throw new Error(`Validation request failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as ValidationResult;
}

export interface RunRequest {
  pipeline: PipelineDefinition;
  input: unknown;
}

/**
 * Stream pipeline execution events. Uses chunked HTTP transfer with NDJSON
 * (one JSON event per line) consumed via a fetch ReadableStream.
 */
export async function* runPipeline(
  pipeline: PipelineDefinition,
  input: unknown,
  signal?: AbortSignal
): AsyncGenerator<ExecutionEvent, void, void> {
  const res = await fetch(`${baseUrl()}/runs`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      accept: 'application/x-ndjson'
    },
    body: JSON.stringify({ pipeline, input } satisfies RunRequest),
    signal
  });

  if (!res.ok || !res.body) {
    throw new Error(`Run request failed: ${res.status} ${res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let newlineIdx: number;
      while ((newlineIdx = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newlineIdx).trim();
        buffer = buffer.slice(newlineIdx + 1);
        if (!line) continue;
        try {
          yield JSON.parse(line) as ExecutionEvent;
        } catch (err) {
          console.warn('Failed to parse runner event line', line, err);
        }
      }
    }
    const tail = buffer.trim();
    if (tail) {
      try {
        yield JSON.parse(tail) as ExecutionEvent;
      } catch (err) {
        console.warn('Failed to parse trailing runner event', tail, err);
      }
    }
  } finally {
    reader.releaseLock();
  }
}
