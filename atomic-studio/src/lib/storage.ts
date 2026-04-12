import { get, set, del, keys } from 'idb-keyval';
import type { PipelineDefinition } from './types';

const PIPELINE_PREFIX = 'pipeline:';
const LAST_KEY = 'meta:last';

export interface StoredPipeline {
  name: string;
  updatedAt: number;
  definition: PipelineDefinition;
}

export async function savePipeline(name: string, definition: PipelineDefinition): Promise<void> {
  const entry: StoredPipeline = { name, updatedAt: Date.now(), definition };
  await set(PIPELINE_PREFIX + name, entry);
  await set(LAST_KEY, name);
}

export async function loadPipeline(name: string): Promise<StoredPipeline | undefined> {
  return await get<StoredPipeline>(PIPELINE_PREFIX + name);
}

export async function deletePipeline(name: string): Promise<void> {
  await del(PIPELINE_PREFIX + name);
  const last = await get<string>(LAST_KEY);
  if (last === name) await del(LAST_KEY);
}

export async function listPipelines(): Promise<string[]> {
  const allKeys = (await keys()) as string[];
  return allKeys
    .filter((k) => typeof k === 'string' && k.startsWith(PIPELINE_PREFIX))
    .map((k) => k.slice(PIPELINE_PREFIX.length))
    .sort();
}

export async function getLastPipelineName(): Promise<string | undefined> {
  return await get<string>(LAST_KEY);
}

export function downloadPipeline(name: string, definition: PipelineDefinition): void {
  const blob = new Blob([JSON.stringify(definition, null, 2)], {
    type: 'application/json'
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${name || 'pipeline'}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function pickPipelineFile(): Promise<PipelineDefinition> {
  return new Promise((resolve, reject) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'application/json,.json';
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) {
        reject(new Error('No file selected'));
        return;
      }
      try {
        const text = await file.text();
        resolve(JSON.parse(text));
      } catch (err) {
        reject(err);
      }
    };
    input.click();
  });
}
