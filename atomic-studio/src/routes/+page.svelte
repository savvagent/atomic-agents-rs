<script lang="ts">
  console.log('[atomic-studio] +page.svelte: starting imports');

  import {
    SvelteFlow,
    Background,
    Controls,
    MiniMap,
    type NodeTypes
  } from '@xyflow/svelte';
  console.log('[atomic-studio] +page.svelte: @xyflow/svelte imported');

  import { onMount } from 'svelte';

  import { pipeline } from '$lib/stores/pipeline.svelte';
  console.log('[atomic-studio] +page.svelte: pipeline store imported', pipeline);

  import StudioNode from '$lib/components/StudioNode.svelte';
  import NodePalette from '$lib/components/NodePalette.svelte';
  import ConfigPanel from '$lib/components/ConfigPanel.svelte';
  import Toolbar from '$lib/components/Toolbar.svelte';
  import SchemaSidebar from '$lib/components/SchemaSidebar.svelte';
  console.log('[atomic-studio] +page.svelte: components imported');

  import { getLastPipelineName, loadPipeline } from '$lib/storage';
  import { NODE_COLORS, type StudioNodeType } from '$lib/types';
  console.log('[atomic-studio] +page.svelte: all imports complete');

  const nodeTypes: NodeTypes = { studioNode: StudioNode as any };

  let flowInstance = $state<any>(null);

  onMount(async () => {
    console.log('[atomic-studio] +page.svelte mounted');
    try {
      const last = await getLastPipelineName();
      console.log('[atomic-studio] last pipeline name:', last);
      if (last) {
        const entry = await loadPipeline(last);
        console.log('[atomic-studio] loaded pipeline entry:', entry);
        if (entry) pipeline.loadPipelineDefinition(entry.definition);
      }
    } catch (err) {
      console.error('[atomic-studio] failed to restore last pipeline', err);
    }

    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'z') {
        e.preventDefault();
        if (e.shiftKey) pipeline.redo();
        else pipeline.undo();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  function handleAddNode(type: StudioNodeType) {
    const viewport = flowInstance?.getViewport?.();
    const x = viewport ? (-viewport.x + 400) / viewport.zoom : 300;
    const y = viewport ? (-viewport.y + 300) / viewport.zoom : 200;
    pipeline.addNode(type, { x, y });
  }

  function handleDrop(event: DragEvent) {
    event.preventDefault();
    const type = event.dataTransfer?.getData('application/studio-node-type') as StudioNodeType;
    if (!type || !flowInstance?.screenToFlowPosition) return;
    const position = flowInstance.screenToFlowPosition({
      x: event.clientX,
      y: event.clientY
    });
    pipeline.addNode(type, position);
  }

  function handleDragOver(event: DragEvent) {
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
  }

  function minimapColor(node: any): string {
    return NODE_COLORS[node?.data?.studioType as StudioNodeType] ?? '#64748b';
  }
</script>

<svelte:head>
  <title>Atomic Studio</title>
</svelte:head>

<div class="flex flex-col h-screen">
  <Toolbar />
  <div class="flex flex-1 overflow-hidden">
    <NodePalette onAddNode={handleAddNode} />

    <div
      class="flex-1 relative"
      role="presentation"
      ondrop={handleDrop}
      ondragover={handleDragOver}
    >
      <SvelteFlow
        bind:nodes={pipeline.nodes}
        bind:edges={pipeline.edges}
        {nodeTypes}
        onnodeclick={({ node }: any) => pipeline.selectNode(node.id)}
        onpaneclick={() => pipeline.selectNode(null)}
        oninit={(instance: any) => (flowInstance = instance)}
        fitView
        deleteKey={['Backspace', 'Delete']}
        defaultEdgeOptions={{
          type: 'smoothstep',
          animated: false,
          style: 'stroke: #64748b; stroke-width: 2;'
        }}
      >
        <Background bgColor="#1a1a2e" gap={20} />
        <Controls />
        <MiniMap nodeColor={minimapColor} />
      </SvelteFlow>
      <SchemaSidebar />
    </div>

    <ConfigPanel />
  </div>
</div>
