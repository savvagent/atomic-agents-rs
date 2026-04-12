console.log('[atomic-studio] pipeline.svelte.ts: module top');

import type { Node, Edge, Connection } from '@xyflow/svelte';
import { addEdge } from '@xyflow/svelte';
console.log('[atomic-studio] pipeline.svelte.ts: imported @xyflow/svelte helpers');

import {
  defaultNodeConfig,
  type StudioNodeType,
  type NodeStatus,
  type PipelineDefinition,
  type ExecutionEvent,
  type PipelineSettings
} from '../types';
console.log('[atomic-studio] pipeline.svelte.ts: imported types');

export interface StudioNodeData {
  studioType: StudioNodeType;
  label: string;
  config: any;
  status: NodeStatus;
  output?: any;
  error?: string;
  [key: string]: unknown;
}

interface HistorySnapshot {
  nodes: Node<StudioNodeData>[];
  edges: Edge[];
}

const DEFAULT_SETTINGS: PipelineSettings = {
  default_provider: 'openai',
  default_model: 'gpt-4o',
  max_concurrent: 4
};

class PipelineStore {
  nodes = $state<Node<StudioNodeData>[]>([]);
  edges = $state<Edge[]>([]);
  schemas = $state<Record<string, any>>({});
  settings = $state<PipelineSettings>({ ...DEFAULT_SETTINGS });
  selectedNodeId = $state<string | null>(null);
  isExecuting = $state(false);

  past = $state<HistorySnapshot[]>([]);
  future = $state<HistorySnapshot[]>([]);

  selectedNode = $derived(
    this.selectedNodeId ? this.nodes.find((n) => n.id === this.selectedNodeId) ?? null : null
  );

  #idCounter = 0;

  #nextId(): string {
    return `node_${++this.#idCounter}`;
  }

  onConnect = (connection: Connection) => {
    this.pushHistory();
    this.edges = addEdge(connection, this.edges);
  };

  addNode(type: StudioNodeType, position: { x: number; y: number }) {
    this.pushHistory();
    const id = this.#nextId();
    const label = type.charAt(0).toUpperCase() + type.slice(1).replace('_', ' ');
    const node: Node<StudioNodeData> = {
      id,
      type: 'studioNode',
      position,
      data: {
        studioType: type,
        label,
        config: defaultNodeConfig(type),
        status: 'idle'
      }
    };
    this.nodes = [...this.nodes, node];
  }

  deleteNode(id: string) {
    this.pushHistory();
    this.nodes = this.nodes.filter((n) => n.id !== id);
    this.edges = this.edges.filter((e) => e.source !== id && e.target !== id);
    if (this.selectedNodeId === id) this.selectedNodeId = null;
  }

  selectNode(id: string | null) {
    this.selectedNodeId = id;
  }

  updateNodeConfig(id: string, config: any) {
    this.nodes = this.nodes.map((n) =>
      n.id === id ? { ...n, data: { ...n.data, config } } : n
    );
  }

  updateNodeLabel(id: string, label: string) {
    this.nodes = this.nodes.map((n) =>
      n.id === id ? { ...n, data: { ...n.data, label } } : n
    );
  }

  addSchema(name: string, schema: any) {
    this.schemas = { ...this.schemas, [name]: schema };
  }

  updateSchema(name: string, schema: any) {
    this.schemas = { ...this.schemas, [name]: schema };
  }

  deleteSchema(name: string) {
    const { [name]: _, ...rest } = this.schemas;
    this.schemas = rest;
  }

  updateSettings(partial: Partial<PipelineSettings>) {
    this.settings = { ...this.settings, ...partial };
  }

  toPipelineDefinition(): PipelineDefinition {
    return {
      schemas: this.schemas,
      nodes: this.nodes.map((n) => ({
        id: n.id,
        type: n.data.studioType,
        config: n.data.config,
        position: [n.position.x, n.position.y]
      })),
      edges: this.edges.map((e) => ({
        source: e.source,
        target: e.target,
        handle: e.sourceHandle ?? undefined
      })),
      settings: this.settings
    };
  }

  loadPipelineDefinition(def: PipelineDefinition) {
    this.#idCounter = 0;
    const nodes: Node<StudioNodeData>[] = def.nodes.map((n) => {
      const numId = parseInt(n.id.replace(/\D/g, '')) || 0;
      if (numId > this.#idCounter) this.#idCounter = numId;
      return {
        id: n.id,
        type: 'studioNode',
        position: { x: n.position[0], y: n.position[1] },
        data: {
          studioType: n.type as StudioNodeType,
          label: n.type.charAt(0).toUpperCase() + n.type.slice(1).replace('_', ' '),
          config: n.config,
          status: 'idle'
        }
      };
    });
    const edges: Edge[] = def.edges.map((e, i) => ({
      id: `e${i}`,
      source: e.source,
      target: e.target,
      sourceHandle: e.handle
    }));

    this.nodes = nodes;
    this.edges = edges;
    this.schemas = def.schemas ?? {};
    this.settings = { ...DEFAULT_SETTINGS, ...(def.settings ?? {}) };
    this.selectedNodeId = null;
    this.past = [];
    this.future = [];
  }

  setExecuting(executing: boolean) {
    this.isExecuting = executing;
  }

  handleExecutionEvent(event: ExecutionEvent) {
    switch (event.event_type) {
      case 'node_started':
        this.nodes = this.nodes.map((n) =>
          n.id === event.node_id ? { ...n, data: { ...n.data, status: 'running' } } : n
        );
        break;
      case 'node_completed':
        this.nodes = this.nodes.map((n) =>
          n.id === event.node_id
            ? { ...n, data: { ...n.data, status: 'completed', output: event.output } }
            : n
        );
        break;
      case 'node_failed':
        this.nodes = this.nodes.map((n) =>
          n.id === event.node_id
            ? { ...n, data: { ...n.data, status: 'failed', error: event.error } }
            : n
        );
        break;
      case 'pipeline_completed':
      case 'pipeline_failed':
        this.isExecuting = false;
        break;
    }
  }

  resetNodeStatuses() {
    this.nodes = this.nodes.map((n) => ({
      ...n,
      data: { ...n.data, status: 'idle', output: undefined, error: undefined }
    }));
  }

  pushHistory() {
    this.past = [...this.past.slice(-50), { nodes: this.nodes, edges: this.edges }];
    this.future = [];
  }

  undo() {
    if (this.past.length === 0) return;
    const prev = this.past[this.past.length - 1];
    this.future = [{ nodes: this.nodes, edges: this.edges }, ...this.future];
    this.past = this.past.slice(0, -1);
    this.nodes = prev.nodes;
    this.edges = prev.edges;
  }

  redo() {
    if (this.future.length === 0) return;
    const next = this.future[0];
    this.past = [...this.past, { nodes: this.nodes, edges: this.edges }];
    this.future = this.future.slice(1);
    this.nodes = next.nodes;
    this.edges = next.edges;
  }
}

console.log('[atomic-studio] pipeline.svelte.ts: instantiating PipelineStore');
export const pipeline = new PipelineStore();
console.log('[atomic-studio] pipeline.svelte.ts: PipelineStore instantiated', pipeline);
