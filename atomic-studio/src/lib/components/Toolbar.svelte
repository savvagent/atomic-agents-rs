<script lang="ts">
  import { pipeline } from '../stores/pipeline.svelte';
  import {
    savePipeline,
    listPipelines,
    loadPipeline,
    downloadPipeline,
    pickPipelineFile
  } from '../storage';
  import { validatePipeline, runPipeline } from '../runner';

  let saving = $state(false);
  let currentName = $state('untitled');
  let savedNames = $state<string[]>([]);

  $effect(() => {
    listPipelines().then((n) => (savedNames = n));
  });

  async function refreshList() {
    savedNames = await listPipelines();
  }

  async function handleSave() {
    saving = true;
    try {
      const name = prompt('Save as:', currentName);
      if (!name) return;
      currentName = name;
      await savePipeline(name, pipeline.toPipelineDefinition());
      await refreshList();
    } finally {
      saving = false;
    }
  }

  async function handleLoad(name: string) {
    const entry = await loadPipeline(name);
    if (entry) {
      pipeline.loadPipelineDefinition(entry.definition);
      currentName = name;
    }
  }

  function handleExport() {
    downloadPipeline(currentName, pipeline.toPipelineDefinition());
  }

  async function handleImport() {
    try {
      const def = await pickPipelineFile();
      pipeline.loadPipelineDefinition(def);
    } catch (err) {
      alert('Import failed: ' + (err as Error).message);
    }
  }

  async function handleValidate() {
    try {
      const result = await validatePipeline(pipeline.toPipelineDefinition());
      if (result.valid) {
        alert('Pipeline is valid!');
      } else {
        alert('Validation errors:\n' + result.errors.map((e) => e.message).join('\n'));
      }
    } catch (err) {
      alert('Validation failed: ' + (err as Error).message);
    }
  }

  async function handleRun() {
    pipeline.resetNodeStatuses();
    pipeline.setExecuting(true);
    try {
      const inputStr = prompt('Enter JSON input:', '{"message": "hello"}');
      if (!inputStr) {
        pipeline.setExecuting(false);
        return;
      }
      const input = JSON.parse(inputStr);
      for await (const event of runPipeline(pipeline.toPipelineDefinition(), input)) {
        pipeline.handleExecutionEvent(event);
      }
    } catch (err) {
      alert('Execution failed: ' + (err as Error).message);
      pipeline.setExecuting(false);
    }
  }
</script>

<div class="flex items-center gap-2 px-4 py-2 bg-panel border-b border-gray-700">
  <span class="text-sm font-bold text-gray-200 mr-4">Atomic Studio</span>

  <input
    bind:value={currentName}
    class="bg-gray-800 border border-gray-600 rounded text-xs text-gray-200 px-2 py-1 w-32 outline-none focus:border-blue-500"
    placeholder="Pipeline name"
  />

  <button class="tb-btn" onclick={handleSave} disabled={saving}>Save</button>

  {#if savedNames.length > 0}
    <select
      class="bg-gray-800 border border-gray-600 rounded text-xs text-gray-200 px-2 py-1 outline-none"
      onchange={(e) => {
        const v = (e.currentTarget as HTMLSelectElement).value;
        if (v) handleLoad(v);
        (e.currentTarget as HTMLSelectElement).value = '';
      }}
    >
      <option value="">Load…</option>
      {#each savedNames as n (n)}
        <option value={n}>{n}</option>
      {/each}
    </select>
  {/if}

  <div class="w-px h-5 bg-gray-600 mx-1"></div>
  <button class="tb-btn" onclick={handleImport}>Import</button>
  <button class="tb-btn" onclick={handleExport}>Export</button>

  <div class="w-px h-5 bg-gray-600 mx-1"></div>
  <button class="tb-btn" onclick={() => pipeline.undo()} disabled={pipeline.past.length === 0}
    >Undo</button
  >
  <button class="tb-btn" onclick={() => pipeline.redo()} disabled={pipeline.future.length === 0}
    >Redo</button
  >

  <div class="w-px h-5 bg-gray-600 mx-1"></div>
  <button class="tb-btn" onclick={handleValidate}>Validate</button>
  <button
    class="tb-btn bg-green-700 hover:bg-green-600"
    onclick={handleRun}
    disabled={pipeline.isExecuting}
  >
    {pipeline.isExecuting ? 'Running…' : 'Run'}
  </button>
</div>

