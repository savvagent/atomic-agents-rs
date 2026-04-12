<script lang="ts">
  import { NODE_COLORS, NODE_LABELS, type StudioNodeType } from '../types';

  interface Props {
    onAddNode: (type: StudioNodeType) => void;
  }

  let { onAddNode }: Props = $props();

  const nodeTypes: StudioNodeType[] = [
    'user_input',
    'agent',
    'tool',
    'router',
    'merge',
    'output'
  ];

  function handleDragStart(event: DragEvent, type: StudioNodeType) {
    if (!event.dataTransfer) return;
    event.dataTransfer.setData('application/studio-node-type', type);
    event.dataTransfer.effectAllowed = 'move';
  }
</script>

<div class="flex flex-col gap-1.5 p-3 bg-panel border-r border-gray-700 w-48">
  <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">Nodes</h3>
  {#each nodeTypes as type (type)}
    <button
      type="button"
      onclick={() => onAddNode(type)}
      ondragstart={(e) => handleDragStart(e, type)}
      draggable="true"
      class="flex items-center gap-2 px-3 py-2 rounded-md text-sm text-gray-200 hover:bg-accent transition-colors text-left"
    >
      <span
        class="w-3 h-3 rounded-full inline-block"
        style="background: {NODE_COLORS[type]}"
      ></span>
      {NODE_LABELS[type]}
    </button>
  {/each}
</div>
