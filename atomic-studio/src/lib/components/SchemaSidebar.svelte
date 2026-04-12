<script lang="ts">
  import SchemaBuilder from './SchemaBuilder.svelte';
  let open = $state(false);
</script>

<button
  class="absolute bottom-4 left-52 z-10 px-3 py-1.5 bg-panel border border-gray-600 rounded text-xs text-gray-300 hover:bg-accent"
  onclick={() => (open = !open)}
>
  {open ? 'Hide Schemas' : 'Schemas'}
</button>

{#if open}
  <div
    class="absolute bottom-12 left-52 z-10 w-72 max-h-96 overflow-y-auto bg-panel border border-gray-700 rounded-lg shadow-xl p-3"
  >
    <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
      Schema Builder
    </h3>
    <SchemaBuilder />
  </div>
{/if}
