<script lang="ts">
  import { pipeline } from '../stores/pipeline.svelte';
  import { NODE_COLORS, type StudioNodeType } from '../types';

  const node = $derived(pipeline.selectedNode);
  const color = $derived(node ? NODE_COLORS[node.data.studioType as StudioNodeType] : '#334155');
  const schemaNames = $derived(Object.keys(pipeline.schemas));

  function updateConfig(patch: any) {
    if (!node) return;
    pipeline.updateNodeConfig(node.id, { ...node.data.config, ...patch });
  }

  function setSystemPrompt(sp: any) {
    updateConfig({ system_prompt: sp });
  }

  function addCondition() {
    if (!node) return;
    const cs = node.data.config.conditions ?? [];
    updateConfig({
      conditions: [...cs, { name: `route_${cs.length}`, field: '', op: 'equals', value: '' }]
    });
  }

  function updateCondition(i: number, patch: any) {
    if (!node) return;
    const cs = node.data.config.conditions ?? [];
    updateConfig({ conditions: cs.map((c: any, idx: number) => (idx === i ? { ...c, ...patch } : c)) });
  }

  function removeCondition(i: number) {
    if (!node) return;
    const cs = node.data.config.conditions ?? [];
    updateConfig({ conditions: cs.filter((_: any, idx: number) => idx !== i) });
  }

  function addPromptItem(key: 'background' | 'steps' | 'output_instructions') {
    if (!node) return;
    const sp = node.data.config.system_prompt ?? {
      background: [],
      steps: [],
      output_instructions: []
    };
    setSystemPrompt({ ...sp, [key]: [...(sp[key] ?? []), ''] });
  }

  function updatePromptItem(
    key: 'background' | 'steps' | 'output_instructions',
    i: number,
    value: string
  ) {
    if (!node) return;
    const sp = node.data.config.system_prompt ?? {
      background: [],
      steps: [],
      output_instructions: []
    };
    const next = [...(sp[key] ?? [])];
    next[i] = value;
    setSystemPrompt({ ...sp, [key]: next });
  }

  function removePromptItem(key: 'background' | 'steps' | 'output_instructions', i: number) {
    if (!node) return;
    const sp = node.data.config.system_prompt ?? {
      background: [],
      steps: [],
      output_instructions: []
    };
    setSystemPrompt({ ...sp, [key]: (sp[key] ?? []).filter((_: any, idx: number) => idx !== i) });
  }
</script>

<div class="w-80 bg-panel border-l border-gray-700 overflow-y-auto">
  {#if !node}
    <div class="p-4 text-gray-400 text-sm">Select a node to configure</div>
  {:else}
    <div
      class="flex items-center justify-between p-3 border-b"
      style="border-bottom-color: {color}"
    >
      <input
        class="bg-transparent text-gray-100 font-semibold text-sm outline-none flex-1 border-b border-transparent focus:border-gray-500"
        value={node.data.label}
        oninput={(e) => pipeline.updateNodeLabel(node.id, (e.currentTarget as HTMLInputElement).value)}
      />
      <button
        class="text-gray-500 hover:text-red-400 ml-2 text-xs"
        onclick={() => {
          pipeline.deleteNode(node.id);
          pipeline.selectNode(null);
        }}
      >
        Delete
      </button>
    </div>

    <div class="p-3 space-y-3">
      {#if node.data.studioType === 'user_input'}
        <div>
          <span class="cfg-label">Input Schema</span>
          <select
            class="input-field"
            value={node.data.config.schema_ref ?? ''}
            onchange={(e) => updateConfig({ schema_ref: (e.currentTarget as HTMLSelectElement).value })}
          >
            <option value="">Select schema…</option>
            {#each schemaNames as name (name)}
              <option value={name}>{name}</option>
            {/each}
          </select>
        </div>
      {:else if node.data.studioType === 'agent'}
        <div class="grid grid-cols-2 gap-2">
          <div>
            <span class="cfg-label">Provider</span>
            <select
              class="input-field"
              value={node.data.config.provider ?? ''}
              onchange={(e) =>
                updateConfig({ provider: (e.currentTarget as HTMLSelectElement).value || undefined })}
            >
              <option value="">Default</option>
              <option value="openai">OpenAI</option>
              <option value="anthropic">Anthropic</option>
              <option value="google">Google</option>
            </select>
          </div>
          <div>
            <span class="cfg-label">Model</span>
            <input
              class="input-field"
              value={node.data.config.model ?? ''}
              placeholder="Default model"
              oninput={(e) =>
                updateConfig({ model: (e.currentTarget as HTMLInputElement).value || undefined })}
            />
          </div>
        </div>

        <div>
          <span class="cfg-label">Input Schema</span>
          <select
            class="input-field"
            value={node.data.config.input_schema ?? ''}
            onchange={(e) => updateConfig({ input_schema: (e.currentTarget as HTMLSelectElement).value })}
          >
            <option value="">Select schema…</option>
            {#each schemaNames as name (name)}
              <option value={name}>{name}</option>
            {/each}
          </select>
        </div>

        <div>
          <span class="cfg-label">Output Schema</span>
          <select
            class="input-field"
            value={node.data.config.output_schema ?? ''}
            onchange={(e) =>
              updateConfig({ output_schema: (e.currentTarget as HTMLSelectElement).value })}
          >
            <option value="">Select schema…</option>
            {#each schemaNames as name (name)}
              <option value={name}>{name}</option>
            {/each}
          </select>
        </div>

        <div>
          <span class="cfg-label">Extraction Mode</span>
          <select
            class="input-field"
            value={node.data.config.extraction_mode ?? 'tools'}
            onchange={(e) =>
              updateConfig({ extraction_mode: (e.currentTarget as HTMLSelectElement).value })}
          >
            <option value="tools">Tools (function calling)</option>
            <option value="json">JSON mode</option>
            <option value="md_json">Markdown JSON</option>
          </select>
        </div>

        <div>
          <span class="cfg-label">Max Retries</span>
          <input
            type="number"
            class="input-field w-20"
            min={0}
            max={10}
            value={node.data.config.max_retries ?? 3}
            oninput={(e) =>
              updateConfig({
                max_retries: parseInt((e.currentTarget as HTMLInputElement).value) || 0
              })}
          />
        </div>

        <div class="space-y-2">
          <span class="cfg-label">System Prompt</span>
          {#each ['background', 'steps', 'output_instructions'] as key (key)}
            {@const items = node.data.config.system_prompt?.[key] ?? []}
            <div>
              <span class="text-xs text-gray-500">{key}</span>
              {#each items as item, i (i)}
                <div class="flex gap-1 mt-1">
                  <input
                    class="input-field flex-1 text-xs"
                    value={item}
                    oninput={(e) =>
                      updatePromptItem(
                        key as any,
                        i,
                        (e.currentTarget as HTMLInputElement).value
                      )}
                  />
                  <button
                    class="text-red-400 hover:text-red-300 text-xs px-1"
                    onclick={() => removePromptItem(key as any, i)}
                  >
                    X
                  </button>
                </div>
              {/each}
              <button
                class="text-xs text-blue-400 hover:text-blue-300 mt-1"
                onclick={() => addPromptItem(key as any)}
              >
                + Add
              </button>
            </div>
          {/each}
        </div>
      {:else if node.data.studioType === 'tool'}
        <div>
          <span class="cfg-label">Tool Type</span>
          <select
            class="input-field"
            value={node.data.config.tool_type ?? 'builtin'}
            onchange={(e) => updateConfig({ tool_type: (e.currentTarget as HTMLSelectElement).value })}
          >
            <option value="builtin">Builtin</option>
            <option value="mcp">MCP</option>
          </select>
        </div>
        <div>
          <span class="cfg-label">Tool Name</span>
          <input
            class="input-field"
            value={node.data.config.name ?? ''}
            oninput={(e) => updateConfig({ name: (e.currentTarget as HTMLInputElement).value })}
          />
        </div>
        {#if node.data.config.tool_type === 'mcp'}
          <div>
            <span class="cfg-label">Endpoint</span>
            <input
              class="input-field"
              value={node.data.config.endpoint ?? ''}
              placeholder="http://localhost:8080"
              oninput={(e) => updateConfig({ endpoint: (e.currentTarget as HTMLInputElement).value })}
            />
          </div>
          <div>
            <span class="cfg-label">Transport</span>
            <select
              class="input-field"
              value={node.data.config.transport ?? 'http_stream'}
              onchange={(e) =>
                updateConfig({ transport: (e.currentTarget as HTMLSelectElement).value })}
            >
              <option value="http_stream">HTTP Stream</option>
              <option value="sse">SSE</option>
              <option value="stdio">Stdio</option>
            </select>
          </div>
        {/if}
      {:else if node.data.studioType === 'router'}
        <span class="cfg-label">Conditions</span>
        {#each node.data.config.conditions ?? [] as cond, i (i)}
          <div class="bg-gray-800 rounded p-2 space-y-1.5 text-xs">
            <div class="flex gap-1">
              <input
                class="input-field flex-1"
                placeholder="Route name"
                value={cond.name}
                oninput={(e) => updateCondition(i, { name: (e.currentTarget as HTMLInputElement).value })}
              />
              <button
                class="text-red-400 hover:text-red-300 px-1"
                onclick={() => removeCondition(i)}
              >
                X
              </button>
            </div>
            <div class="flex gap-1">
              <input
                class="input-field flex-1"
                placeholder="Field"
                value={cond.field}
                oninput={(e) => updateCondition(i, { field: (e.currentTarget as HTMLInputElement).value })}
              />
              <select
                class="input-field w-24"
                value={cond.op}
                onchange={(e) => updateCondition(i, { op: (e.currentTarget as HTMLSelectElement).value })}
              >
                <option value="equals">equals</option>
                <option value="contains">contains</option>
                <option value="greater_than">{'>'}</option>
                <option value="less_than">{'<'}</option>
                <option value="matches">matches</option>
                <option value="exists">exists</option>
              </select>
              {#if cond.op !== 'exists'}
                <input
                  class="input-field flex-1"
                  placeholder="Value"
                  value={typeof cond.value === 'string' ? cond.value : JSON.stringify(cond.value)}
                  oninput={(e) => {
                    let val: any = (e.currentTarget as HTMLInputElement).value;
                    try {
                      val = JSON.parse(val);
                    } catch {
                      // ignore
                    }
                    updateCondition(i, { value: val });
                  }}
                />
              {/if}
            </div>
          </div>
        {/each}
        <button class="text-xs text-blue-400 hover:text-blue-300" onclick={addCondition}>
          + Add condition
        </button>
        <div>
          <span class="cfg-label">Default Route</span>
          <input
            class="input-field"
            value={node.data.config.default_route ?? 'default'}
            oninput={(e) =>
              updateConfig({ default_route: (e.currentTarget as HTMLInputElement).value })}
          />
        </div>
      {:else if node.data.studioType === 'merge'}
        <div>
          <span class="cfg-label">Strategy</span>
          <select
            class="input-field"
            value={node.data.config.strategy ?? 'first_ready'}
            onchange={(e) => updateConfig({ strategy: (e.currentTarget as HTMLSelectElement).value })}
          >
            <option value="first_ready">First Ready</option>
            <option value="wait_all">Wait All</option>
            <option value="merge_objects">Merge Objects</option>
          </select>
        </div>
        <div>
          <span class="cfg-label">Output Schema (optional)</span>
          <select
            class="input-field"
            value={node.data.config.output_schema ?? ''}
            onchange={(e) =>
              updateConfig({
                output_schema: (e.currentTarget as HTMLSelectElement).value || undefined
              })}
          >
            <option value="">Select schema…</option>
            {#each schemaNames as name (name)}
              <option value={name}>{name}</option>
            {/each}
          </select>
        </div>
      {:else if node.data.studioType === 'output'}
        <div>
          <span class="cfg-label">Output Schema (optional)</span>
          <select
            class="input-field"
            value={node.data.config.schema_ref ?? ''}
            onchange={(e) =>
              updateConfig({
                schema_ref: (e.currentTarget as HTMLSelectElement).value || undefined
              })}
          >
            <option value="">Select schema…</option>
            {#each schemaNames as name (name)}
              <option value={name}>{name}</option>
            {/each}
          </select>
        </div>
      {/if}

      {#if node.data.output}
        <div class="mt-4">
          <span class="cfg-label">Last Output</span>
          <pre
            class="text-xs bg-gray-900 p-2 rounded overflow-auto max-h-40 text-green-400">{JSON.stringify(
              node.data.output,
              null,
              2
            )}</pre>
        </div>
      {/if}

      {#if node.data.error}
        <div class="mt-4">
          <span class="block text-xs font-semibold text-red-400 mb-1">Error</span>
          <pre class="text-xs bg-gray-900 p-2 rounded text-red-300">{node.data.error}</pre>
        </div>
      {/if}
    </div>
  {/if}
</div>

