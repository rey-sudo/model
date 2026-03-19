<template>
  <div
    class="h-screen box-border grid grid-cols-1 md:grid-cols-[200px_1fr_600px]"
  >
    <aside
      class="bg-[var(--color-surface)] p-4 overflow-y-auto border-r border-[var(--color-border)]"
    >
      <TabTrainExplorer />
    </aside>

    <main class="bg-[var(--color-surface)] p-4 box-border">
      <TabTrainCenter />
    </main>

    <aside
      class="bg-[var(--color-surface)] p-4  box-border border-l border-[var(--color-border)]"
    >
      <TabTrainBlock />
    </aside>
  </div>
</template>

<script setup>

</script>

<style lang="css" scoped></style>
