export type StudioNodeType =
  | 'user_input'
  | 'agent'
  | 'tool'
  | 'router'
  | 'merge'
  | 'output';

export interface SchemaDefinition {
  [key: string]: any;
}

export interface SystemPromptConfig {
  background: string[];
  steps: string[];
  output_instructions: string[];
}

export interface AgentNodeConfig {
  provider?: string;
  model?: string;
  input_schema: string;
  output_schema: string;
  extraction_mode: string;
  system_prompt: SystemPromptConfig;
  max_retries: number;
  extra_params: Record<string, any>;
}

export interface UserInputConfig {
  schema_ref: string;
}

export interface ToolNodeConfig {
  tool_type: string;
  name: string;
  endpoint?: string;
  transport?: string;
}

export interface RouteCondition {
  name: string;
  field: string;
  op: string;
  value: any;
}

export interface RouterConfig {
  conditions: RouteCondition[];
  default_route: string;
}

export interface MergeConfig {
  strategy: string;
  output_schema?: string;
}

export interface OutputConfig {
  schema_ref?: string;
}

export interface PipelineNodeDef {
  id: string;
  type: string;
  config: any;
  position: [number, number];
}

export interface PipelineEdgeDef {
  source: string;
  target: string;
  handle?: string;
}

export interface PipelineSettings {
  default_provider: string;
  default_model: string;
  max_concurrent: number;
}

export interface PipelineDefinition {
  schemas: Record<string, any>;
  nodes: PipelineNodeDef[];
  edges: PipelineEdgeDef[];
  settings: PipelineSettings;
}

export interface ValidationIssue {
  node_id?: string;
  message: string;
}

export interface ValidationResult {
  valid: boolean;
  errors: ValidationIssue[];
  warnings: ValidationIssue[];
}

export type ExecutionEvent =
  | { event_type: 'node_started'; node_id: string }
  | { event_type: 'node_completed'; node_id: string; output: any }
  | { event_type: 'node_failed'; node_id: string; error: string }
  | { event_type: 'pipeline_completed'; outputs: Record<string, any> }
  | { event_type: 'pipeline_failed'; error: string };

export type NodeStatus = 'idle' | 'running' | 'completed' | 'failed';

export const NODE_COLORS: Record<StudioNodeType, string> = {
  user_input: '#8b5cf6',
  agent: '#3b82f6',
  tool: '#f59e0b',
  router: '#e94560',
  merge: '#06b6d4',
  output: '#22c55e'
};

export const NODE_LABELS: Record<StudioNodeType, string> = {
  user_input: 'Input',
  agent: 'Agent',
  tool: 'Tool',
  router: 'Router',
  merge: 'Merge',
  output: 'Output'
};

export const NODE_ICONS: Record<StudioNodeType, string> = {
  user_input: '\u25B6',
  agent: '\u2699',
  tool: '\u2692',
  router: '\u2B95',
  merge: '\u2B8C',
  output: '\u2B24'
};

export function defaultNodeConfig(type: StudioNodeType): any {
  switch (type) {
    case 'user_input':
      return { schema_ref: '' };
    case 'agent':
      return {
        input_schema: '',
        output_schema: '',
        extraction_mode: 'tools',
        system_prompt: { background: [], steps: [], output_instructions: [] },
        max_retries: 3,
        extra_params: {}
      };
    case 'tool':
      return { tool_type: 'builtin', name: '' };
    case 'router':
      return { conditions: [], default_route: 'default' };
    case 'merge':
      return { strategy: 'first_ready' };
    case 'output':
      return {};
  }
}
