<template>
  <div
    class="h-screen box-border grid grid-cols-1 md:grid-cols-[200px_1fr_600px]"
  >
    <aside
      class="bg-[var(--color-surface)] p-4 overflow-y-auto border-r border-[var(--color-border)]"
    >
      <TabTrainExplorer />
    </aside>

    <main class="bg-[var(--color-surface)] p-4 grid gap-4 box-border">
      <ChartMatrix3d :points="misDatos" , :target="puntoCentral" />
      <Console />
    </main>

    <aside
      class="bg-[var(--color-surface)] p-4 overflow-y-auto border-l border-[var(--color-border)]"
    >
      <TabTrainBlock />
    </aside>
  </div>
</template>

<script setup>
const misDatos = {
  x: [
    8298.85, 3047.98, 2934.95, 5120.4, 1200.15, 7500.6, 4300.2, 9100.0, 2500.5,
    6800.3, 1500.9, 5900.4, 3100.75,
  ],
  y: [
    8759.67, 3781.44, 1110.77, 4500.3, 2100.8, 7800.9, 3200.45, 9500.1, 1800.6,
    6200.25, 2900.7, 5300.85, 4100.2,
  ],
  z: [
    62.87, 887.32, 7063.16, 1200.45, 5400.12, 300.55, 8500.9, 150.2, 4200.75,
    2100.3, 9800.6, 3300.4, 1100.95,
  ],
  labels: [
    "territory",
    "area",
    "residents",
    "sector A",
    "sector B",
    "zone alpha",
    "zone beta",
    "central",
    "outskirts",
    "urban",
    "rural",
    "industrial",
    "commercial",
  ],
};

const puntoCentral = {
  x: 4200.24,
  y: 6149.22,
  z: 2799.93,
  label: "bank",
};
</script>

<style lang="css" scoped></style>
