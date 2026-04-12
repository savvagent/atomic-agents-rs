<script lang="ts">
  import { Handle, Position, type NodeProps } from '@xyflow/svelte';
  import { NODE_COLORS, NODE_LABELS, NODE_ICONS, type StudioNodeType } from '../types';
  import type { StudioNodeData } from '../stores/pipeline.svelte';

  type Props = NodeProps & { data: StudioNodeData };
  let { data, selected }: Props = $props();

  const color = $derived(NODE_COLORS[data.studioType as StudioNodeType]);
  const statusClass = $derived(
    data.status === 'running'
      ? 'node-status-running'
      : data.status === 'completed'
        ? 'node-status-completed'
        : data.status === 'failed'
          ? 'node-status-failed'
          : ''
  );
  const hasInput = $derived(data.studioType !== 'user_input');
  const hasOutput = $derived(data.studioType !== 'output');
  const conditions = $derived(
    data.studioType === 'router' ? (data.config?.conditions ?? []) : []
  );
</script>

<div class="studio-node {selected ? 'selected' : ''} {statusClass}">
  {#if hasInput}
    <Handle type="target" position={Position.Left} style="background: {color}" />
  {/if}

  <div class="node-header" style="border-bottom-color: {color}">
    <span style="color: {color}">{NODE_ICONS[data.studioType as StudioNodeType]}</span>
    <span class="text-gray-100 text-sm">{data.label}</span>
    <span
      class="ml-auto text-xs px-1.5 py-0.5 rounded"
      style="background: {color}33; color: {color}"
    >
      {NODE_LABELS[data.studioType as StudioNodeType]}
    </span>
  </div>

  <div class="node-body">
    {#if data.studioType === 'agent'}
      <div>{data.config?.model || 'default model'}</div>
      <div class="text-xs mt-1 truncate max-w-[200px]">
        {data.config?.system_prompt?.background?.[0] || 'No prompt'}
      </div>
    {:else if data.studioType === 'user_input'}
      <div>Schema: {data.config?.schema_ref || 'none'}</div>
    {:else if data.studioType === 'tool'}
      <div>{data.config?.name || 'No tool selected'}</div>
    {:else if data.studioType === 'router'}
      <div>{conditions.length} routes</div>
    {:else if data.studioType === 'merge'}
      <div>Strategy: {data.config?.strategy || 'first_ready'}</div>
    {:else if data.studioType === 'output'}
      <div>{data.config?.schema_ref || 'any'}</div>
    {/if}
  </div>

  {#if hasOutput}
    {#if data.studioType === 'router'}
      {#each conditions as cond, i (cond.name)}
        <Handle
          type="source"
          position={Position.Right}
          id={cond.name}
          style="background: {color}; top: {((i + 1) / (conditions.length + 1)) * 100}%"
        />
      {/each}
    {:else}
      <Handle type="source" position={Position.Right} style="background: {color}" />
    {/if}
  {/if}
</div>
