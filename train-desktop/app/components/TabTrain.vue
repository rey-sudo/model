<template>
  <div
    class="h-screen box-border grid grid-cols-1 md:grid-cols-[200px_1fr_550px] "
  >
    <aside
      class="bg-[var(--color-surface)] p-4 overflow-y-auto border-r border-[var(--color-border)]"
    ></aside>

    <main class="p-4 grid gap-4">
      <ChartMatrix3d :points="misDatos" , :target="puntoCentral" />
      <Console />
    </main>

    <aside
      class="bg-[var(--color-surface)] p-4 overflow-y-auto border-l border-[var(--color-border)]"
    ></aside>
  </div>
</template>

<script setup>
const misDatos = {
  x: [8298.85, 3047.98, 2934.95],
  y: [8759.67, 3781.44, 1110.77],
  z: [62.87, 887.32, 7063.16],
  labels: ["vehicle", "wheels", "gas"],
};

const puntoCentral = {
  x: 4200.24,
  y: 6149.22,
  z: 2799.93,
  label: "car",
};
</script>

<style lang="css" scoped></style>
