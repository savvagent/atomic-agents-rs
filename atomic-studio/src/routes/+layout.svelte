<script lang="ts">
  import '../app.css';
  import { onMount } from 'svelte';

  console.log('[atomic-studio] +layout.svelte module evaluated');
  let { children } = $props();

  onMount(() => {
    console.log('[atomic-studio] +layout.svelte mounted');
  });
</script>

{@render children?.()}
