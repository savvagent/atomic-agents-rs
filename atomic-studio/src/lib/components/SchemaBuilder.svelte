<script lang="ts">
  import { pipeline } from '../stores/pipeline.svelte';

  let selectedSchema = $state<string | null>(null);
  let newName = $state('');

  const schemaNames = $derived(Object.keys(pipeline.schemas));
  const currentSchema = $derived(
    selectedSchema ? pipeline.schemas[selectedSchema] : null
  );

  function handleCreate() {
    const name = newName.trim();
    if (!name || pipeline.schemas[name]) return;
    pipeline.addSchema(name, { type: 'object', properties: {}, required: [] });
    selectedSchema = name;
    newName = '';
  }

  function handleDelete(name: string) {
    pipeline.deleteSchema(name);
    if (selectedSchema === name) selectedSchema = null;
  }

  function fields(schema: any) {
    const properties = schema?.properties ?? {};
    const required: string[] = schema?.required ?? [];
    return Object.entries(properties).map(([name, def]: [string, any]) => ({
      name,
      type: def.type ?? 'string',
      description: def.description ?? '',
      required: required.includes(name)
    }));
  }

  function updateField(index: number, updates: Partial<{ name: string; type: string; required: boolean }>) {
    if (!selectedSchema) return;
    const schema = pipeline.schemas[selectedSchema];
    const fs = fields(schema);
    const old = fs[index];
    const next = { ...old, ...updates };
    const newProps: Record<string, any> = { ...(schema.properties ?? {}) };

    if (updates.name && updates.name !== old.name) {
      delete newProps[old.name];
    }
    newProps[next.name] = {
      type: next.type,
      ...(next.description ? { description: next.description } : {})
    };
    const newRequired = fs
      .map((f, i) => (i === index ? next : f))
      .filter((f) => f.required)
      .map((f) => f.name);

    pipeline.updateSchema(selectedSchema, {
      ...schema,
      properties: newProps,
      required: newRequired
    });
  }

  function addField() {
    if (!selectedSchema) return;
    const schema = pipeline.schemas[selectedSchema];
    const properties = schema?.properties ?? {};
    const fieldName = `field_${Object.keys(properties).length}`;
    pipeline.updateSchema(selectedSchema, {
      ...schema,
      properties: { ...properties, [fieldName]: { type: 'string' } }
    });
  }

  function removeField(fieldName: string) {
    if (!selectedSchema) return;
    const schema = pipeline.schemas[selectedSchema];
    const { [fieldName]: _, ...rest } = schema.properties ?? {};
    pipeline.updateSchema(selectedSchema, {
      ...schema,
      properties: rest,
      required: (schema.required ?? []).filter((r: string) => r !== fieldName)
    });
  }
</script>

<div class="space-y-3">
  <div class="flex gap-2">
    <input
      class="input-field flex-1"
      placeholder="New schema name…"
      bind:value={newName}
      onkeydown={(e) => e.key === 'Enter' && handleCreate()}
    />
    <button
      class="px-3 py-1.5 bg-accent text-white text-xs rounded hover:bg-blue-700"
      onclick={handleCreate}
    >
      Create
    </button>
  </div>

  <div class="space-y-1">
    {#each schemaNames as name (name)}
      <div
        class="flex items-center justify-between px-2 py-1.5 rounded cursor-pointer text-sm {selectedSchema ===
        name
          ? 'bg-accent text-white'
          : 'text-gray-300 hover:bg-gray-800'}"
        role="button"
        tabindex="0"
        onclick={() => (selectedSchema = name)}
        onkeydown={(e) => e.key === 'Enter' && (selectedSchema = name)}
      >
        <span>{name}</span>
        <button
          class="text-red-400 hover:text-red-300 text-xs"
          onclick={(e) => {
            e.stopPropagation();
            handleDelete(name);
          }}
        >
          Del
        </button>
      </div>
    {/each}
  </div>

  {#if selectedSchema && currentSchema}
    <div class="border border-gray-700 rounded p-2 space-y-2">
      <h4 class="text-xs font-semibold text-gray-300">{selectedSchema}</h4>

      {#each fields(currentSchema) as field, i (i)}
        <div class="flex gap-1 items-start">
          <input
            class="input-field flex-1 text-xs"
            placeholder="Field name"
            value={field.name}
            oninput={(e) => updateField(i, { name: (e.currentTarget as HTMLInputElement).value })}
          />
          <select
            class="input-field w-20 text-xs"
            value={field.type}
            onchange={(e) => updateField(i, { type: (e.currentTarget as HTMLSelectElement).value })}
          >
            <option value="string">string</option>
            <option value="number">number</option>
            <option value="integer">integer</option>
            <option value="boolean">boolean</option>
            <option value="array">array</option>
            <option value="object">object</option>
          </select>
          <label class="flex items-center gap-1 text-xs text-gray-400">
            <input
              type="checkbox"
              checked={field.required}
              onchange={(e) =>
                updateField(i, { required: (e.currentTarget as HTMLInputElement).checked })}
            />
            Req
          </label>
          <button
            class="text-red-400 hover:text-red-300 text-xs px-1"
            onclick={() => removeField(field.name)}
          >
            X
          </button>
        </div>
      {/each}

      <button class="text-xs text-blue-400 hover:text-blue-300" onclick={addField}>
        + Add field
      </button>

      <details class="mt-2">
        <summary class="text-xs text-gray-500 cursor-pointer">JSON Preview</summary>
        <pre
          class="text-xs bg-gray-900 p-2 rounded mt-1 overflow-auto max-h-32 text-gray-400">{JSON.stringify(
            currentSchema,
            null,
            2
          )}</pre>
      </details>
    </div>
  {/if}
</div>
