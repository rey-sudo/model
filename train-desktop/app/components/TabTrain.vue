<template>
  <div
    class="h-screen box-border grid grid-cols-1 md:grid-cols-[200px_1fr_600px]"
  >
    <aside
      class="bg-[var(--color-surface)] p-4 overflow-y-auto border-r border-[var(--color-border)]"
    ></aside>

    <main class="bg-[var(--color-surface)]  p-4 grid gap-4 box-border">
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
  x: [8298.85, 3047.98, 2934.95, 5120.40, 1200.15, 7500.60, 4300.20, 9100.00, 2500.50, 6800.30, 1500.90, 5900.40, 3100.75],
  y: [8759.67, 3781.44, 1110.77, 4500.30, 2100.80, 7800.90, 3200.45, 9500.10, 1800.60, 6200.25, 2900.70, 5300.85, 4100.20],
  z: [62.87, 887.32, 7063.16, 1200.45, 5400.12, 300.55, 8500.90, 150.20, 4200.75, 2100.30, 9800.60, 3300.40, 1100.95],
  labels: [
    "territory", "area", "residents", 
    "sector A", "sector B", "zone alpha", 
    "zone beta", "central", "outskirts", 
    "urban", "rural", "industrial", "commercial"
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
